import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from mslb import fakedata
from mslb.config import from_dict
from mslb.schemas import fc_column, fc_wide
from mslb.stats import describe, fdr_bh, mann_whitney, responder, run_analyses, spearman

RNG = np.random.default_rng(42)


def test_spearman_matches_scipy_and_drops_missing_pairs():
    x = RNG.normal(size=30)
    y = 0.5 * x + RNG.normal(size=30)
    ref = sps.spearmanr(x, y)
    got = spearman(x, y)
    assert got["estimate"] == pytest.approx(ref.statistic)
    assert got["p"] == pytest.approx(ref.pvalue)
    assert got["n"] == 30 and got["status"] == "ok"
    x2, y2 = np.append(x, [np.nan, 1.0]), np.append(y, [1.0, np.nan])
    again = spearman(x2, y2)
    assert again["n"] == 30 and again["estimate"] == pytest.approx(ref.statistic)


def test_spearman_interval_brackets_rho_and_is_seeded():
    x = RNG.normal(size=40)
    y = x + RNG.normal(size=40)
    a, b, c = spearman(x, y, seed=1), spearman(x, y, seed=1), spearman(x, y, seed=2)
    assert a["ci_low"] < a["estimate"] < a["ci_high"]
    assert a["ci_low"] > 0                      # a strong effect at n=40 excludes zero
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])


def test_spearman_handles_ties_like_scipy():
    x = np.array([0, 0, 0, 0, 1, 2, 3, 5, 8, 8], float)       # zero-inflated, like tract loads
    y = np.array([1, 2, 2, 3, 3, 4, 6, 5, 9, 7], float)
    assert spearman(x, y)["estimate"] == pytest.approx(sps.spearmanr(x, y).statistic)


def test_spearman_skips_rather_than_inventing_a_result():
    few = spearman([1, 2, 3], [1, 2, 3])
    assert few["status"].startswith("skipped") and np.isnan(few["estimate"]) and np.isnan(few["p"])
    flat = spearman(np.zeros(10), np.arange(10))
    assert flat["status"] == "skipped: no variation" and np.isnan(flat["p"])


def test_mann_whitney_matches_scipy():
    a, b = RNG.normal(1, 1, 12), RNG.normal(0, 1, 9)
    ref = sps.mannwhitneyu(a, b, alternative="two-sided")
    got = mann_whitney(a, b)
    assert got["statistic"] == pytest.approx(ref.statistic)
    assert got["p"] == pytest.approx(ref.pvalue)
    assert (got["n_group1"], got["n_group0"], got["n"]) == (12, 9, 21)
    assert got["estimate"] == pytest.approx(2 * ref.statistic / (12 * 9) - 1)
    assert got["ci_low"] <= got["estimate"] <= got["ci_high"]
    assert got["shift_ci_low"] <= got["shift"] <= got["shift_ci_high"]


def test_rank_biserial_direction_and_extremes():
    low, high = np.arange(6.0), np.arange(6.0) + 100
    assert mann_whitney(high, low)["estimate"] == 1.0
    assert mann_whitney(low, high)["estimate"] == -1.0
    assert mann_whitney(high, low)["shift"] == pytest.approx(100.0)
    same = mann_whitney([1, 2, 3, 4], [1, 2, 3, 4])
    assert same["estimate"] == pytest.approx(0.0) and same["shift"] == 0.0


def test_mann_whitney_skips_small_or_flat_groups():
    assert mann_whitney([1, 2], [3, 4, 5, 6, 7])["status"].startswith("skipped")
    assert mann_whitney([0, 0, 0], [0, 0, 0])["status"] == "skipped: no variation"
    drops_nan = mann_whitney([1, 2, 3, np.nan], [4, 5, 6])
    assert drops_nan["n_group1"] == 3


def test_fdr_matches_statsmodels_and_a_hand_calculation():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    assert fdr_bh(p) == pytest.approx(multipletests(p, method="fdr_bh")[1])
    # smallest p times m over rank: 0.001 * 5 / 1
    assert fdr_bh(p)[0] == pytest.approx(0.005)
    assert fdr_bh([0.03])[0] == pytest.approx(0.03)          # a family of one is unchanged


def test_fdr_leaves_missing_p_missing_and_out_of_the_count():
    q = fdr_bh([0.01, np.nan, 0.04])
    assert np.isnan(q[1])
    assert q[[0, 2]] == pytest.approx(multipletests([0.01, 0.04], method="fdr_bh")[1])
    assert np.isnan(fdr_bh([np.nan, np.nan])).all()


def test_describe_and_suppression():
    d = describe([1, 2, 3, 4, np.nan])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert (d["q1"], d["median"], d["q3"]) == (1.75, 2.5, 3.25)
    assert "min" not in d and "max" not in d                 # extremes are individual values
    small = describe([1, 2], min_cell_n=3)
    assert small["n"] == 2 and small["suppressed"] and np.isnan(small["mean"])


def subject_table(n=40, seed=0):
    """A per-subject table shaped like the pipeline's, with the fake tables joined on."""
    ids = fakedata.default_ids(n)
    rng = np.random.default_rng(seed)
    burden = rng.lognormal(1, 1, n)
    clinical, fc = fakedata.make_fake(ids, seed=seed, burden=dict(zip(ids, burden)))
    imaging = pd.DataFrame({
        "subject_id": ids,
        "lesion_volume_ml": burden, "lesion_volume_norm_ml": burden * rng.uniform(0.9, 1.1, n),
        "lesion_count": rng.integers(1, 40, n),
        "tract_ATR_ml": rng.exponential(0.3, n), "tract_ATR_norm_ml": rng.exponential(0.3, n),
        "tract_SLF_ml": rng.exponential(0.3, n), "tract_SLF_norm_ml": rng.exponential(0.3, n),
    })
    return imaging.merge(clinical, on="subject_id").merge(fc_wide(fc), on="subject_id")


def test_planted_effects_are_found_and_survive_fdr():
    cfg = from_dict({})
    res = run_analyses(subject_table(), cfg)
    q1 = res[(res.question == "q1_total_burden_fc") & (res.exposure == "total lesion volume")]
    planted = q1[(q1.state == "full") & (q1.connection == fakedata.PLANTED_CONNECTION)].iloc[0]
    assert planted["estimate"] < -0.5 and planted["q"] < 0.05
    others = q1[(q1.state == "full") & (q1.connection != fakedata.PLANTED_CONNECTION)]
    assert (others["q"] > 0.05).all()
    nbss = res[(res.question == "q4_symptom_severity") & (res.outcome == "nbss_total")
               & (res.exposure == "total lesion volume")].iloc[0]
    assert nbss["estimate"] > 0.5 and nbss["p"] < 0.001 and nbss["role"] == "primary"


def test_fdr_runs_within_exposure_and_state_across_connections_only():
    cfg = from_dict({})
    res = run_analyses(subject_table(), cfg)
    n_conn = len(fakedata.connection_names())
    fc_rows = res[res.question.isin(["q1_total_burden_fc", "q2_tract_load_fc"])]
    # q1: volume + count, q2: two tracts. each over three states
    assert fc_rows["family"].nunique() == (2 + 2) * 3
    assert (fc_rows.groupby("family").size() == n_conn).all()
    for _, family in fc_rows.groupby("family"):
        assert family["q"].to_numpy() == pytest.approx(fdr_bh(family["p"]))
    rest = res[~res.question.isin(["q1_total_burden_fc", "q2_tract_load_fc"])]
    assert rest["q"].isna().all() and (rest["p_adjustment"] == "none").all()


def test_question_layout_and_q5_off_by_default():
    res = run_analyses(subject_table(), from_dict({}))
    assert set(res.question) == {"q1_total_burden_fc", "q2_tract_load_fc",
                                 "q3_detrusor_overactivity", "q4_symptom_severity"}
    q3 = res[res.question == "q3_detrusor_overactivity"]
    assert list(q3.outcome) == ["total lesion volume", "ATR lesion volume", "SLF lesion volume"]
    assert (q3.test == "mann_whitney").all() and (q3.n_group1 + q3.n_group0 == q3.n).all()
    q4 = res[res.question == "q4_symptom_severity"]
    assert len(q4) == 3 * 4
    q2 = res[res.question == "q2_tract_load_fc"]
    assert set(q2.exposure) == {"ATR lesion volume", "SLF lesion volume"}


def test_normalized_or_raw_volume_is_chosen_by_the_config():
    table = subject_table()
    target = fc_column("full", fakedata.PLANTED_CONNECTION)

    def rho(cfg):
        res = run_analyses(table, cfg)
        row = res[(res.question == "q1_total_burden_fc") & (res.exposure == "total lesion volume")
                  & (res.state == "full") & (res.connection == fakedata.PLANTED_CONNECTION)]
        return row["estimate"].iloc[0]

    assert rho(from_dict({})) == pytest.approx(
        sps.spearmanr(table["lesion_volume_norm_ml"], table[target]).statistic)
    assert rho(from_dict({"stats": {"volume_scale": "raw"}})) == pytest.approx(
        sps.spearmanr(table["lesion_volume_ml"], table[target]).statistic)


def test_q5_uses_the_active_arm_and_the_configured_threshold():
    cfg = from_dict({"analyses": {"q5_treatment_response": True},
                     "treatment_response": {"min_nbss_improvement": 7}})
    table = pd.DataFrame({
        "arm": ["active", "active", "active", "sham", "active", None],
        "nbss_total": [30, 30, 30, 30, np.nan, 30],
        "nbss_total_post": [23, 24, 40, 10, 20, 10],
    })
    got = responder(table, cfg)
    assert got.iloc[0] == 1          # fell by exactly 7: a response
    assert got.iloc[1] == 0          # fell by 6
    assert got.iloc[2] == 0          # got worse
    assert got.iloc[3:].isna().all()  # sham, missing baseline, unknown arm: not in the test
    res = run_analyses(subject_table(60), cfg)
    q5 = res[res.question == "q5_treatment_response"]
    assert len(q5) == 3 and (q5.n < 60).all()


def test_a_table_without_optional_columns_still_runs():
    table = subject_table().drop(columns=["do_present", "tract_SLF_norm_ml", "diary_voids_per_day"])
    res = run_analyses(table, from_dict({}))
    assert "q3_detrusor_overactivity" not in set(res.question)
    assert "SLF lesion volume" not in set(res.exposure)
    assert "diary_voids_per_day" not in set(res.outcome)


def test_small_cohort_reports_skipped_tests_instead_of_dropping_them():
    res = run_analyses(subject_table(5), from_dict({}))
    assert len(res) > 0
    assert res["status"].str.startswith("skipped").all()
    assert res["p"].isna().all() and res["estimate"].isna().all()


def test_fc_tests_use_only_subjects_scanned_in_both_states():
    table = subject_table(30)
    conn = fakedata.PLANTED_CONNECTION
    full, empty = fc_column("full", conn), fc_column("empty", conn)
    table.loc[:4, empty] = np.nan                       # five subjects have no empty-bladder value
    table.loc[:4, fc_column("change", conn)] = np.nan

    def n_for(cfg, state):
        res = run_analyses(table, cfg)
        row = res[(res.question == "q1_total_burden_fc") & (res.exposure == "total lesion volume")
                  & (res.state == state) & (res.connection == conn)]
        return int(row["n"].iloc[0])

    strict = from_dict({})
    assert n_for(strict, "full") == 25 and n_for(strict, "empty") == 25 and n_for(strict, "change") == 25
    loose = from_dict({"stats": {"fc_require_both_states": False}})
    assert n_for(loose, "full") == 30 and n_for(loose, "empty") == 25


def test_a_supplied_change_value_counts_without_the_two_states():
    table = subject_table(30)
    conn = "seed01__roi01"
    table[fc_column("full", conn)] = np.nan             # exported as differences only
    table[fc_column("empty", conn)] = np.nan
    res = run_analyses(table, from_dict({}))
    rows = res[(res.question == "q1_total_burden_fc") & (res.exposure == "total lesion volume")
               & (res.connection == conn)].set_index("state")
    assert rows.loc["change", "n"] == 30 and rows.loc["change", "status"] == "ok"
    assert rows.loc["full", "status"].startswith("skipped")
