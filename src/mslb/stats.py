"""The tests behind each question, and the one function that runs them all.

Rank-based throughout: Spearman for two continuous variables, Mann-Whitney for two
groups. Every test is two-sided. Benjamini-Hochberg is applied across functional
connectivity outcomes only, within one exposure and one bladder state at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from mslb.config import Config

STATES = ("full", "empty", "change")
SYMPTOM_OUTCOMES = [
    ("nbss_total", "primary"),
    ("diary_voids_per_day", "secondary"),
    ("diary_urgency_per_day", "secondary"),
    ("diary_incontinence_per_day", "secondary"),
]


def _rank_rows(a: np.ndarray) -> np.ndarray:
    return sps.rankdata(a, axis=1)


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def spearman(x, y, n_boot: int = 2000, seed: int = 0, min_n: int = 6) -> dict:
    """Spearman rho with a percentile bootstrap 95% interval.

    Pairs with a missing value are dropped. Too few pairs, or a variable with no
    variation, gives status 'skipped' and no numbers rather than a made-up result.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(len(x))
    out = {"test": "spearman", "estimate_name": "rho", "n": n,
           "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "statistic": np.nan, "p": np.nan, "status": "ok"}
    if n < min_n:
        out["status"] = f"skipped: n below {min_n}"
        return out
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        out["status"] = "skipped: no variation"
        return out
    res = sps.spearmanr(x, y)
    out["estimate"] = out["statistic"] = float(res.statistic)
    out["p"] = float(res.pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = _row_pearson(_rank_rows(x[idx]), _rank_rows(y[idx]))
    boot = boot[np.isfinite(boot)]
    if len(boot) >= n_boot // 2:
        out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return out


def mann_whitney(group1, group0, n_boot: int = 2000, seed: int = 0, min_n: int = 6) -> dict:
    """Mann-Whitney U for group1 against group0.

    estimate is the rank-biserial correlation: +1 when every group1 value is above
    every group0 value, -1 the other way round, 0 for no separation. shift is the
    Hodges-Lehmann estimate, the median of all group1 minus group0 differences, in
    the units of the variable, with a bootstrap 95% interval.
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group0, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n0 = int(len(a)), int(len(b))
    out = {"test": "mann_whitney", "estimate_name": "rank_biserial", "n": n1 + n0,
           "n_group1": n1, "n_group0": n0,
           "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "shift": np.nan, "shift_ci_low": np.nan, "shift_ci_high": np.nan,
           "statistic": np.nan, "p": np.nan, "status": "ok"}
    if n1 + n0 < min_n or min(n1, n0) < 3:
        out["status"] = f"skipped: needs {min_n} in total and 3 per group"
        return out
    if np.ptp(np.concatenate([a, b])) == 0:
        out["status"] = "skipped: no variation"
        return out
    res = sps.mannwhitneyu(a, b, alternative="two-sided")
    u1 = float(res.statistic)
    out["statistic"], out["p"] = u1, float(res.pvalue)
    out["estimate"] = 2 * u1 / (n1 * n0) - 1
    out["shift"] = float(np.median(a[:, None] - b[None, :]))

    rng = np.random.default_rng(seed)
    ia = rng.integers(0, n1, size=(n_boot, n1))
    ib = rng.integers(0, n0, size=(n_boot, n0))
    rb, hl = np.empty(n_boot), np.empty(n_boot)
    # resamples x n1 x n0 differences, a slab at a time so a big cohort does not eat the memory
    step = max(1, 4_000_000 // (n1 * n0))
    for lo in range(0, n_boot, step):
        hi = min(lo + step, n_boot)
        diffs = a[ia[lo:hi]][:, :, None] - b[ib[lo:hi]][:, None, :]
        wins = (diffs > 0).sum(axis=(1, 2)) + 0.5 * (diffs == 0).sum(axis=(1, 2))
        rb[lo:hi] = 2 * wins / (n1 * n0) - 1
        hl[lo:hi] = np.median(diffs.reshape(hi - lo, -1), axis=1)
    out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(rb, [2.5, 97.5]))
    out["shift_ci_low"], out["shift_ci_high"] = (float(v) for v in np.percentile(hl, [2.5, 97.5]))
    return out


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values. Missing p-values stay missing and are not counted."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def describe(values, min_cell_n: int = 1) -> dict:
    """n, mean, SD, median and quartiles. Below min_cell_n only n is given."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {"n": int(len(v)), "mean": np.nan, "sd": np.nan,
           "median": np.nan, "q1": np.nan, "q3": np.nan, "suppressed": False}
    if len(v) == 0:
        return out
    if len(v) < min_cell_n:
        out["suppressed"] = True
        return out
    out["mean"] = float(v.mean())
    out["sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    out["median"], out["q1"], out["q3"] = (float(x) for x in np.percentile(v, [50, 25, 75]))
    return out


# ---- which columns play which part ----------------------------------------------------

def volume_column(name: str, cfg: Config) -> str:
    """Column for one volume exposure: 'total' or a tract name."""
    suffix = "norm_ml" if cfg.stats.volume_scale == "normalized" else "ml"
    return f"lesion_volume_{suffix}" if name == "total" else f"tract_{name}_{suffix}"


def volume_exposures(cfg: Config, total: bool = True) -> list[tuple[str, str]]:
    """(label, column) for total lesion volume and each primary tract."""
    names = (["total"] if total else []) + list(cfg.tracts.primary)
    return [("total lesion volume" if n == "total" else f"{n} lesion volume",
             volume_column(n, cfg)) for n in names]


def fc_outcomes(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(column, state, connection) for every FC column in the subject table."""
    out = []
    for col in table.columns:
        if col.startswith("fc__"):
            _, state, connection = col.split("__", 2)
            out.append((col, state, connection))
    return out


def _n_nonzero(series: pd.Series) -> int:
    v = pd.to_numeric(series, errors="coerce")
    return int((v > 0).sum())


def _missing(table: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if c not in table.columns]


def _row(question, role, exposure, outcome, result, **extra) -> dict:
    row = {"question": question, "role": role, "exposure": exposure, "outcome": outcome,
           "state": "", "connection": "", "family": "", "q": np.nan, "p_adjustment": "none"}
    row.update(result)
    row.update(extra)
    return row


def fc_values(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The FC columns as they enter the tests.

    The FC questions are about subjects scanned with a full and an empty bladder. With
    fc_require_both_states on, a subject's full or empty value for a connection only
    counts when the other state is there too. The change column needs no such rule: it
    is either computed from both states or supplied as a difference.
    """
    cols = [c for c, _, _ in fc_outcomes(table)]
    out = table[cols].apply(pd.to_numeric, errors="coerce")
    if not cfg.stats.fc_require_both_states:
        return out
    for col, state, connection in fc_outcomes(table):
        other = {"full": "empty", "empty": "full"}.get(state)
        if other is None:
            continue
        partner = f"fc__{other}__{connection}"
        if partner in out.columns:
            out[col] = out[col].where(table[partner].notna())
        else:
            out[col] = np.nan
    return out


def _fc_block(table, cfg, question, exposures) -> list[dict]:
    rows = []
    st = cfg.stats
    fc = fc_values(table, cfg)
    for role, label, col in exposures:
        if col not in table.columns:
            continue
        for state in STATES:
            family = f"{question} | {label} | {state}"
            block = []
            for fc_col, fc_state, connection in fc_outcomes(table):
                if fc_state != state:
                    continue
                res = spearman(table[col], fc[fc_col], st.bootstrap_resamples, st.seed, st.min_n)
                block.append(_row(question, role, label, f"FC {state}: {connection}", res,
                                  state=state, connection=connection, family=family,
                                  n_nonzero_exposure=_n_nonzero(table[col]),
                                  p_adjustment="fdr_bh across connections"))
            for row, q in zip(block, fdr_bh([r["p"] for r in block])):
                row["q"] = float(q)
            rows.extend(block)
    return rows


def _group_block(table, cfg, question, group: pd.Series, labels: tuple[str, str]) -> list[dict]:
    rows = []
    st = cfg.stats
    for label, col in volume_exposures(cfg):
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        res = mann_whitney(values[group == 1], values[group == 0],
                           st.bootstrap_resamples, st.seed, st.min_n)
        rows.append(_row(question, "primary" if col.startswith("lesion_volume") else "secondary",
                         f"{labels[0]} vs {labels[1]}", label, res,
                         n_nonzero_exposure=_n_nonzero(values)))
    return rows


def responder(table: pd.DataFrame, cfg: Config) -> pd.Series:
    """1 responder, 0 non-responder, missing for anyone outside the active arm.

    Response is an improvement (a fall) in NBSS total of at least the configured
    number of points between baseline and the post-treatment score.
    """
    threshold = cfg.treatment_response.min_nbss_improvement
    improvement = pd.to_numeric(table["nbss_total"], errors="coerce") - pd.to_numeric(
        table["nbss_total_post"], errors="coerce")
    out = pd.Series(np.nan, index=table.index)
    active = table["arm"].astype("string").str.lower().eq("active").fillna(False) & improvement.notna()
    out[active] = (improvement[active] >= threshold).astype(float)
    return out


def run_analyses(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Every enabled test, one row each. Tests that cannot run are kept, marked skipped."""
    rows: list[dict] = []
    on = cfg.analyses

    if on.q1_total_burden_fc:
        exposures = [("primary", "total lesion volume", volume_column("total", cfg)),
                     ("secondary", "lesion count", "lesion_count")]
        rows += _fc_block(table, cfg, "q1_total_burden_fc", exposures)

    if on.q2_tract_load_fc:
        exposures = [("primary", label, col) for label, col in volume_exposures(cfg, total=False)]
        rows += _fc_block(table, cfg, "q2_tract_load_fc", exposures)

    if on.q3_detrusor_overactivity and "do_present" in table.columns:
        group = pd.to_numeric(table["do_present"], errors="coerce")
        rows += _group_block(table, cfg, "q3_detrusor_overactivity", group,
                             ("DO present", "DO absent"))

    if on.q4_symptom_severity:
        st = cfg.stats
        for label, col in volume_exposures(cfg):
            if col not in table.columns:
                continue
            for outcome, role in SYMPTOM_OUTCOMES:
                if outcome not in table.columns:
                    continue
                res = spearman(table[col], pd.to_numeric(table[outcome], errors="coerce"),
                               st.bootstrap_resamples, st.seed, st.min_n)
                rows.append(_row("q4_symptom_severity", role, label, outcome, res,
                                 n_nonzero_exposure=_n_nonzero(table[col])))

    if on.q5_treatment_response and not _missing(table, ["arm", "nbss_total", "nbss_total_post"]):
        rows += _group_block(table, cfg, "q5_treatment_response", responder(table, cfg),
                             ("responder", "non-responder"))

    columns = ["question", "role", "exposure", "outcome", "state", "connection", "test",
               "n", "n_group1", "n_group0", "n_nonzero_exposure", "estimate_name", "estimate",
               "ci_low", "ci_high", "shift", "shift_ci_low", "shift_ci_high", "statistic",
               "p", "q", "p_adjustment", "family", "status"]
    frame = pd.DataFrame(rows)
    for c in columns:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame[columns]
