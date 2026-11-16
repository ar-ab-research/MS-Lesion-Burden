"""The shareable group summary: counts, descriptives and test results, no subject rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mslb import stats
from mslb.fakedata import FAKE_COLUMN
from mslb.config import Config

TITLES = {
    "q1_total_burden_fc": "Q1. Total lesion burden and functional connectivity",
    "q2_tract_load_fc": "Q2. Tract lesion load and functional connectivity",
    "q3_detrusor_overactivity": "Q3. Lesion burden by detrusor overactivity",
    "q4_symptom_severity": "Q4. Lesion burden and symptom severity",
    "q5_treatment_response": "Q5. Lesion burden by treatment response",
}
FC_QUESTIONS = ("q1_total_burden_fc", "q2_tract_load_fc")


def _imaging_columns(table: pd.DataFrame, cfg: Config) -> list[str]:
    cols = ["lesion_volume_ml", "lesion_volume_norm_ml", "lesion_count", "head_scale", "etiv_ml"]
    for name in list(cfg.tracts.primary) + list(cfg.tracts.secondary):
        cols += [f"tract_{name}_ml", f"tract_{name}_norm_ml"]
    return [c for c in cols if c in table.columns]


def _clinical_columns(table: pd.DataFrame) -> list[str]:
    cols = [name for name, _ in stats.SYMPTOM_OUTCOMES] + ["nbss_total_post"]
    return [c for c in cols if c in table.columns]


def descriptives(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """n, mean, SD, median and quartiles: overall, and by group where a question compares groups."""
    min_n = cfg.report.min_cell_n
    rows = []

    def add(group: str, frame: pd.DataFrame, cols: list[str]) -> None:
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            d = stats.describe(values, min_n)
            d["n_nonzero"] = int((values > 0).sum()) if not d["suppressed"] else np.nan
            rows.append({"group": group, "variable": col, **d})

    imaging = _imaging_columns(table, cfg)
    fc_cols = [c for c, _, _ in stats.fc_outcomes(table)]
    add("all", table, imaging + _clinical_columns(table) + fc_cols)

    if cfg.analyses.q3_detrusor_overactivity and "do_present" in table.columns:
        do = pd.to_numeric(table["do_present"], errors="coerce")
        add("DO present", table[do == 1], imaging)
        add("DO absent", table[do == 0], imaging)
    if cfg.analyses.q5_treatment_response and {"arm", "nbss_total", "nbss_total_post"} <= set(table.columns):
        resp = stats.responder(table, cfg)
        add("responder", table[resp == 1], imaging)
        add("non-responder", table[resp == 0], imaging)
    return pd.DataFrame(rows)


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    # three significant figures, written out in full: 1183, not 1.18e+03
    if abs(value) >= 10 ** digits:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


def _p(value) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _describe_rows(desc: pd.DataFrame, group: str, variables: list[str]) -> list[list[str]]:
    rows = []
    for var in variables:
        hit = desc[(desc["group"] == group) & (desc["variable"] == var)]
        if hit.empty:
            continue
        d = hit.iloc[0]
        if d["suppressed"]:
            rows.append([var, _fmt(int(d["n"])), "withheld", "", ""])
        else:
            rows.append([var, _fmt(int(d["n"])), f"{_fmt(d['mean'])} ± {_fmt(d['sd'])}",
                         f"{_fmt(d['median'])} ({_fmt(d['q1'])} to {_fmt(d['q3'])})",
                         _fmt(int(d["n_nonzero"]))])
    return rows


def _fc_section(tests: pd.DataFrame, alpha: float) -> list[str]:
    lines = _table(
        ["Exposure", "State", "Connections tested", f"q < {alpha:g}", "Smallest q"],
        [[exposure, state, _fmt(int(fam["status"].eq("ok").sum())),
          _fmt(int((fam["q"] < alpha).sum())), _p(fam["q"].min())]
         for (exposure, state), fam in tests.groupby(["exposure", "state"], sort=False)],
    )
    hits = tests[tests["q"] < alpha].sort_values("q")
    shown, caption = hits, f"Connections with q < {alpha:g}:"
    if hits.empty:
        shown = tests[tests["status"] == "ok"].nsmallest(5, "p")
        caption = (f"No connection reached q < {alpha:g}. For orientation only, the five smallest "
                   "p-values (none is a finding):")
    if not shown.empty:
        lines += ["", caption, ""]
        lines += _table(
            ["Exposure", "State", "Connection", "n", "rho (95% CI)", "p", "q"],
            [[r.exposure, r.state, r.connection, _fmt(int(r.n)),
              f"{_fmt(r.estimate)} ({_fmt(r.ci_low)} to {_fmt(r.ci_high)})", _p(r.p), _p(r.q)]
             for r in shown.itertuples()],
        )
    return lines


def _corr_section(tests: pd.DataFrame) -> list[str]:
    return _table(
        ["Exposure", "Outcome", "Role", "n", "rho (95% CI)", "p (unadjusted)"],
        [[r.exposure, r.outcome, r.role, _fmt(int(r.n)),
          f"{_fmt(r.estimate)} ({_fmt(r.ci_low)} to {_fmt(r.ci_high)})" if r.status == "ok" else r.status,
          _p(r.p)] for r in tests.itertuples()],
    )


def _group_section(tests: pd.DataFrame) -> list[str]:
    return _table(
        ["Comparison", "Outcome", "n (group 1 / group 0)", "Rank-biserial r (95% CI)",
         "Median shift, mL (95% CI)", "p (unadjusted)"],
        [[r.exposure, r.outcome, f"{_fmt(int(r.n_group1))} / {_fmt(int(r.n_group0))}",
          f"{_fmt(r.estimate)} ({_fmt(r.ci_low)} to {_fmt(r.ci_high)})" if r.status == "ok" else r.status,
          f"{_fmt(r.shift)} ({_fmt(r.shift_ci_low)} to {_fmt(r.shift_ci_high)})" if r.status == "ok" else "",
          _p(r.p)] for r in tests.itertuples()],
    )


def uses_fake_data(table: pd.DataFrame) -> bool:
    """True when any row came from `mslb make-fake-data`."""
    if FAKE_COLUMN not in table.columns:
        return False
    return bool((pd.to_numeric(table[FAKE_COLUMN], errors="coerce") == 1).any())


FAKE_BANNER = ("**FAKE DATA. The clinical and FC tables behind this summary were made up by "
               "`mslb make-fake-data` to test the pipeline. Nothing below is a finding.**")


def markdown(table: pd.DataFrame, tests: pd.DataFrame, desc: pd.DataFrame, cfg: Config,
             manifest: dict) -> str:
    counts, alpha = manifest["counts"], cfg.stats.alpha
    scale = ("adjusted for head size (raw volume × head-size factor)"
             if cfg.stats.volume_scale == "normalized" else "raw, not adjusted for head size")
    lines = [
        "# Group summary",
        "",
        *([FAKE_BANNER, ""] if uses_fake_data(table) else []),
        "Group-level results only. This file holds no subject IDs and no per-subject values,",
        "so it can be shared. How to read each number: `docs/interpreting-results.md`.",
        "",
        f"- Scans processed: {counts['processed']} of {counts['scans']}"
        + (f" ({counts['failed']} failed)" if counts["failed"] else ""),
        f"- With clinical data: {counts['with_clinical']}. With FC data: {counts['with_fc']}.",
        f"- Registration QC failures (no head-size or tract values): {counts['registration_qc_failed']}",
        f"- Lesion volumes in the tests are {scale}.",
        f"- All tests are two-sided. FC results carry Benjamini-Hochberg q-values, computed across",
        "  connections within one exposure and one bladder state. Other p-values are unadjusted.",
        f"- Group statistics are withheld for groups smaller than {cfg.report.min_cell_n}.",
        f"- mslb {manifest['versions']['mslb']}, MindGlide {manifest['versions']['mindglide']}, "
        f"config {manifest['config_digest']}, finished {manifest['finished']}.",
        "",
        "## Cohort",
        "",
    ]
    variables = _imaging_columns(table, cfg) + _clinical_columns(table)
    lines += _table(["Variable", "n", "Mean ± SD", "Median (IQR)", "n above zero"],
                    _describe_rows(desc, "all", variables))
    lines.append("")

    for question, title in TITLES.items():
        block = tests[tests["question"] == question]
        if block.empty:
            continue
        lines += [f"## {title}", ""]
        skipped = int(block["status"].ne("ok").sum())
        if question in FC_QUESTIONS:
            lines += _fc_section(block, alpha)
        elif question == "q4_symptom_severity":
            lines += _corr_section(block)
        else:
            groups = ("DO present", "DO absent") if question == "q3_detrusor_overactivity" else (
                "responder", "non-responder")
            lines += _group_section(block)
            imaging = [c for c in _imaging_columns(table, cfg)
                       if c in {stats.volume_column(n, cfg) for n in ["total", *cfg.tracts.primary]}]
            for g in groups:
                rows = _describe_rows(desc, g, imaging)
                if rows:
                    lines += ["", f"{g}:", ""]
                    lines += _table(["Variable", "n", "Mean ± SD", "Median (IQR)", "n above zero"], rows)
            lines += ["", f"Group 1 is {groups[0]}, group 0 is {groups[1]}. A positive rank-biserial r "
                          "or shift means group 1 has the larger volumes."]
        if skipped:
            lines += ["", f"{skipped} test(s) could not be run (too few subjects or no variation) "
                          "and are listed as skipped in the csv."]
        lines.append("")
    lines += ["Every test, including all FC connections, is in `group_summary.csv`. "
              "Descriptives for every variable are in `descriptives.csv`.", ""]
    return "\n".join(lines)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def write(table: pd.DataFrame, tests: pd.DataFrame, cfg: Config, manifest: dict) -> list[Path]:
    """Write the shareable files. Returns their paths."""
    out = cfg.paths.shareable_dir
    out.mkdir(parents=True, exist_ok=True)
    desc = descriptives(table, cfg)
    paths = [out / "group_summary.md", out / "group_summary.csv", out / "descriptives.csv",
             out / "group_summary.json"]
    paths[0].write_text(markdown(table, tests, desc, cfg, manifest), encoding="utf-8")
    tests.to_csv(paths[1], index=False)
    desc.to_csv(paths[2], index=False)
    paths[3].write_text(json.dumps(
        {"fake_data": uses_fake_data(table), "run": manifest,
         "descriptives": _records(desc), "tests": _records(tests)}, indent=2),
        encoding="utf-8")
    return paths
