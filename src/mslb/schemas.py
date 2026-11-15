"""The two input tables: what columns they hold, and checking a file against that.

COLUMNS is the single description of the schema. The templates and the data
dictionary in templates/ are written from it, and the loaders check against it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mslb.config import Config
from mslb.errors import PipelineError


class SchemaError(PipelineError):
    """An input table does not match its schema. The message lists every problem."""


@dataclass(frozen=True)
class Column:
    table: str            # clinical | fc
    name: str
    kind: str             # id | number | binary | choice | text
    description: str
    needed_for: str = ""  # analyses that need it, comma separated. empty: always
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()

    @property
    def allowed(self) -> str:
        if self.kind == "binary":
            return "0 or 1"
        if self.choices:
            return " | ".join(self.choices)
        if self.low is not None and self.high is not None:
            return f"{self.low:g} to {self.high:g}"
        if self.low is not None:
            return f">= {self.low:g}"
        return ""


FC_STATES = ("full", "empty", "change")

COLUMNS = [
    Column("clinical", "subject_id", "id",
           "Coded subject ID. Must match the T1 file or folder name. No names, dates or record numbers."),
    Column("clinical", "do_present", "binary",
           "Detrusor overactivity on urodynamics: 1 present, 0 absent.",
           needed_for="q3_detrusor_overactivity"),
    Column("clinical", "nbss_total", "number",
           "Neurogenic Bladder Symptom Score, total, at baseline.",
           needed_for="q4_symptom_severity, q5_treatment_response", low=0, high=74),
    Column("clinical", "diary_voids_per_day", "number",
           "Bladder diary: mean voids per day at baseline.",
           needed_for="q4_symptom_severity", low=0),
    Column("clinical", "diary_urgency_per_day", "number",
           "Bladder diary: mean urgency episodes per day at baseline.",
           needed_for="q4_symptom_severity", low=0),
    Column("clinical", "diary_incontinence_per_day", "number",
           "Bladder diary: mean incontinence episodes per day at baseline.",
           needed_for="q4_symptom_severity", low=0),
    Column("clinical", "arm", "choice",
           "Treatment arm. Only read when q5 is on.",
           needed_for="q5_treatment_response", choices=("active", "sham")),
    Column("clinical", "nbss_total_post", "number",
           "NBSS total right after treatment. Only read when q5 is on.",
           needed_for="q5_treatment_response", low=0, high=74),
    Column("fc", "subject_id", "id", "Coded subject ID, as in the clinical table."),
    Column("fc", "state", "choice",
           "Bladder state of the scan. 'change' rows are optional: without them the "
           "pipeline computes full minus empty itself.",
           choices=FC_STATES),
    Column("fc", "connection", "text",
           "Any label for the FC outcome, for example a pair of regions. The same label "
           "must be used for every subject and state."),
    Column("fc", "fc_z", "number", "Functional connectivity value, Fisher z."),
]


def columns(table: str) -> list[Column]:
    return [c for c in COLUMNS if c.table == table]


def template(table: str) -> pd.DataFrame:
    """An empty table with the right header."""
    return pd.DataFrame(columns=[c.name for c in columns(table)])


def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [{"table": c.table, "column": c.name, "type": c.kind, "allowed": c.allowed,
          "needed_for": c.needed_for or "always", "description": c.description}
         for c in COLUMNS]
    )


def write_templates(folder: Path) -> list[Path]:
    """Write the two empty templates and the data dictionary."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, frame in [("clinical_template.csv", template("clinical")),
                        ("fc_template.csv", template("fc")),
                        ("data_dictionary.csv", data_dictionary())]:
        frame.to_csv(folder / name, index=False, lineterminator="\n")
        out.append(folder / name)
    return out


def _read(path: Path) -> pd.DataFrame:
    try:
        # everything as text first, so ids like 001 keep their zeros and bad cells can be named
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001  pandas raises several unrelated types here
        raise SchemaError(f"{path}: cannot be read as csv ({exc})") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.apply(lambda col: col.str.strip())


def _check_values(frame: pd.DataFrame, spec: Column, problems: list[str], label: str) -> pd.Series:
    """Convert one column to its type. Blank cells become missing. Bad cells are reported."""
    raw = frame[spec.name]
    blank = raw.eq("") | raw.str.lower().isin(["na", "nan", "n/a", "null"])

    def report(bad: pd.Series, why: str) -> None:
        if bad.any():
            rows = ", ".join(str(i + 2) for i in np.flatnonzero(bad.to_numpy())[:5])
            more = f" and {int(bad.sum()) - 5} more" if bad.sum() > 5 else ""
            problems.append(f"{label}: column {spec.name} {why} (file row {rows}{more})")

    if spec.kind in ("id", "text"):
        report(blank, "is empty")
        return raw
    if spec.kind == "choice":
        lowered = raw.str.lower()
        report(~blank & ~lowered.isin(spec.choices), f"must be one of {spec.allowed}")
        return lowered.where(~blank, other=pd.NA)
    values = pd.to_numeric(raw.where(~blank), errors="coerce")
    report(~blank & (values.isna() | ~np.isfinite(values)), "must be a number")
    if spec.kind == "binary":
        report(values.notna() & ~values.isin([0, 1]), "must be 0 or 1")
    if spec.low is not None:
        report(values < spec.low, f"is below {spec.low:g}")
    if spec.high is not None:
        report(values > spec.high, f"is above {spec.high:g}")
    return values


def _load(path: Path, table: str, required: set[str]) -> pd.DataFrame:
    path = Path(path)
    label = path.name
    frame = _read(path)
    specs = {c.name: c for c in columns(table)}
    problems = []
    missing = [c for c in specs if c in required and c not in frame.columns]
    if missing:
        problems.append(f"{label}: missing column(s) {', '.join(missing)}")
    dupes = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
    if dupes:
        problems.append(f"{label}: column(s) appear twice: {', '.join(dupes)}")
    if problems:
        raise SchemaError("\n".join(problems))
    out = pd.DataFrame(index=frame.index)
    for name, spec in specs.items():
        if name in frame.columns:
            out[name] = _check_values(frame, spec, problems, label)
    # columns the schema does not know are carried along untouched, never tested
    for name in frame.columns:
        if name not in specs:
            out[name] = frame[name]
    if problems:
        raise SchemaError("\n".join(problems))
    return out


def clinical_required(cfg: Config) -> set[str]:
    """Columns the enabled analyses need. A column for a disabled analysis may be absent."""
    on = set(cfg.enabled_analyses())
    need = {"subject_id"}
    for c in columns("clinical"):
        wanted = {a.strip() for a in c.needed_for.split(",") if a.strip()}
        if wanted & on:
            need.add(c.name)
    return need


def load_clinical(path: Path, cfg: Config) -> pd.DataFrame:
    frame = _load(path, "clinical", clinical_required(cfg))
    dup = frame["subject_id"][frame["subject_id"].duplicated()].unique()
    if len(dup):
        raise SchemaError(f"{Path(path).name}: subject_id appears more than once: {', '.join(dup[:5])}")
    return frame.reset_index(drop=True)


def load_fc(path: Path) -> pd.DataFrame:
    frame = _load(path, "fc", {c.name for c in columns("fc")})
    key = ["subject_id", "state", "connection"]
    dup = frame[frame.duplicated(key, keep=False)]
    if len(dup):
        first = dup.iloc[0]
        raise SchemaError(
            f"{Path(path).name}: {len(dup)} rows repeat a subject, state and connection "
            f"(first: {first.subject_id}, {first.state}, {first.connection})"
        )
    return frame.reset_index(drop=True)


def fc_column(state: str, connection: str) -> str:
    return f"fc__{state}__{connection}"


def fc_wide(fc: pd.DataFrame) -> pd.DataFrame:
    """One row per subject, one column per state and connection.

    change = full minus empty, computed wherever both exist. A supplied change row
    wins over the computed value, so values exported as differences can be used as is.
    """
    wide = fc.pivot(index="subject_id", columns=["state", "connection"], values="fc_z")
    connections = sorted(fc["connection"].unique())
    out = pd.DataFrame(index=wide.index)

    def get(state: str, conn: str) -> pd.Series:
        if (state, conn) in wide.columns:
            return wide[(state, conn)]
        return pd.Series(np.nan, index=wide.index)

    for conn in connections:
        full, empty, given = get("full", conn), get("empty", conn), get("change", conn)
        out[fc_column("full", conn)] = full
        out[fc_column("empty", conn)] = empty
        out[fc_column("change", conn)] = given.where(given.notna(), full - empty)
    return out.dropna(axis=1, how="all").reset_index()
