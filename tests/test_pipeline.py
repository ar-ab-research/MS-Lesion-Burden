import json

import numpy as np
import pandas as pd
import pytest

from mslb import atlas, fakedata, icv, pipeline
from mslb.cli import main
from mslb.config import from_dict, load
from mslb.errors import PipelineError

IDS = [f"case{i:02d}" for i in range(24)]


def imaging_row(sid, rng, reg_ok=True):
    lesion = float(rng.lognormal(1, 1))
    row = {"subject_id": sid, "lesion_volume_ml": lesion, "lesion_count": int(rng.integers(1, 40)),
           "lesion_volume_unfiltered_ml": lesion * 1.02, "labelled_volume_ml": 1400.0,
           "head_scale": float(rng.uniform(1.1, 1.5)), "reg_similarity": 0.8, "reg_ok": reg_ok,
           "etiv_ml": 1300.0, "icv_method": "affine_scaling"}
    for name in ("ATR", "SLF", "ATR_L", "ATR_R", "SLF_L", "SLF_R", "SLF_temporal"):
        row[f"tract_{name}_ml"] = float(rng.exponential(0.2))
        row[f"tract_{name}_size_ml"] = 20.0
        row[f"tract_{name}_pct"] = 1.0
    return row


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Scans (empty files), fake tables, and stand-ins for the slow imaging steps."""
    config = tmp_path / "config.yaml"
    config.write_text("stats: {bootstrap_resamples: 200}\n")
    cfg = load(config)
    for sid in IDS:
        scan = cfg.paths.t1_dir / f"{sid}_T1w.nii.gz"
        scan.parent.mkdir(parents=True, exist_ok=True)
        scan.write_bytes(b"")
    rng = np.random.default_rng(0)
    rows = {sid: imaging_row(sid, rng) for sid in IDS}
    burden = {sid: rows[sid]["lesion_volume_ml"] * rows[sid]["head_scale"] for sid in IDS}
    clinical, fc = fakedata.make_fake(IDS, seed=0, burden=burden)
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    fc.to_csv(cfg.paths.fc_csv, index=False)

    def fake_process(sid, t1, cfg, force, template_ml):
        if rows[sid] is None:
            raise PipelineError(f"{sid}: unreadable scan")
        return dict(rows[sid])

    monkeypatch.setattr(pipeline, "process_subject", fake_process)
    monkeypatch.setattr(atlas, "template_paths", lambda cfg: {"t1": None, "mask": None})
    monkeypatch.setattr(atlas, "jhu_paths", lambda cfg, log=print: {})
    monkeypatch.setattr(icv, "mask_volume_ml", lambda path: 1800.0)
    return config, cfg, rows


def test_run_writes_local_and_shareable_outputs(project):
    config, cfg, _ = project
    assert main(["run", "-c", str(config)]) == 0
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv")
    assert len(subjects) == 24
    assert {"lesion_volume_norm_ml", "tract_ATR_norm_ml", "nbss_total", "do_present"} <= set(subjects)
    assert any(c.startswith("fc__change__") for c in subjects)
    row = subjects.iloc[0]
    assert row["lesion_volume_norm_ml"] == pytest.approx(row["lesion_volume_ml"] * row["head_scale"])
    for name in ("group_summary.md", "group_summary.csv", "descriptives.csv", "group_summary.json"):
        assert (cfg.paths.shareable_dir / name).is_file()
    assert (cfg.paths.local_dir / "run_manifest.json").is_file()


def test_shareable_files_hold_no_subject_ids_or_subject_values(project):
    config, cfg, rows = project
    main(["run", "-c", str(config)])
    for path in cfg.paths.shareable_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "case" not in text, path.name                     # no ids
        assert "subject_id" not in text, path.name
        # no row of the per-subject table either: spot-check one subject's exact volume
        assert f"{rows['case03']['lesion_volume_ml']:.6f}"[:9] not in text, path.name
    desc = pd.read_csv(cfg.paths.shareable_dir / "descriptives.csv")
    assert not {"min", "max"} & set(desc.columns)


def test_summary_reports_the_planted_effects(project):
    config, cfg, _ = project
    main(["run", "-c", str(config)])
    tests = pd.read_csv(cfg.paths.shareable_dir / "group_summary.csv")
    hit = tests[(tests.question == "q1_total_burden_fc") & (tests.exposure == "total lesion volume")
                & (tests.state == "full") & (tests.connection == fakedata.PLANTED_CONNECTION)].iloc[0]
    assert hit["q"] < 0.05
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    for heading in ("## Cohort", "## Q1.", "## Q2.", "## Q3.", "## Q4."):
        assert heading in md
    assert "## Q5." not in md
    assert fakedata.PLANTED_CONNECTION in md
    summary = json.loads((cfg.paths.shareable_dir / "group_summary.json").read_text())
    assert summary["run"]["counts"]["processed"] == 24
    assert "paths" not in summary["run"]["settings"]             # local folder names stay local


def test_a_failed_subject_is_left_out_and_counted_not_fatal(project, capsys):
    config, cfg, rows = project
    rows["case05"] = None
    assert main(["run", "-c", str(config)]) == 0
    assert "case05  FAILED" in capsys.readouterr().out
    assert len(pd.read_csv(cfg.paths.local_dir / "subjects.csv")) == 23
    manifest = json.loads((cfg.paths.local_dir / "run_manifest.json").read_text())
    assert manifest["counts"]["failed"] == 1 and "case05" in manifest["failed_subjects"]
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "23 of 24" in md and "1 failed" in md


def test_registration_qc_failure_blanks_head_size_and_tracts_but_keeps_raw_volume(project):
    config, cfg, rows = project
    rows["case07"]["reg_ok"] = False
    main(["run", "-c", str(config)])
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv").set_index("subject_id")
    bad = subjects.loc["case07"]
    assert bad["lesion_volume_ml"] > 0
    assert np.isnan(bad["lesion_volume_norm_ml"]) and np.isnan(bad["tract_ATR_ml"])
    assert np.isnan(bad["head_scale"])
    tests = pd.read_csv(cfg.paths.shareable_dir / "group_summary.csv")
    q4 = tests[(tests.question == "q4_symptom_severity") & (tests.outcome == "nbss_total")]
    assert (q4["n"] == 23).all()


def test_small_groups_are_withheld_from_the_summary(project):
    config, cfg, _ = project
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    clinical["do_present"] = 0
    clinical.loc[:1, "do_present"] = 1                    # a group of two, under min_cell_n 3
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    main(["run", "-c", str(config)])
    desc = pd.read_csv(cfg.paths.shareable_dir / "descriptives.csv")
    small = desc[desc.group == "DO present"]
    assert (small["n"] == 2).all() and small["suppressed"].all() and small["mean"].isna().all()
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "withheld" in md
    tests = pd.read_csv(cfg.paths.shareable_dir / "group_summary.csv")
    q3 = tests[tests.question == "q3_detrusor_overactivity"]
    assert q3["status"].str.startswith("skipped").all()


def test_q5_appears_only_when_turned_on(project):
    config, cfg, _ = project
    config.write_text("stats: {bootstrap_resamples: 200}\n"
                      "analyses: {q5_treatment_response: true}\n"
                      "treatment_response: {min_nbss_improvement: 6}\n")
    main(["run", "-c", str(config)])
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "## Q5." in md
    tests = pd.read_csv(cfg.paths.shareable_dir / "group_summary.csv")
    assert (tests.question == "q5_treatment_response").sum() == 3


def test_bad_table_stops_the_run_before_any_imaging(project, capsys):
    config, cfg, rows = project
    cfg.paths.fc_csv.write_text("subject_id,state,connection,fc_z\ncase00,resting,a,0.1\n")
    called = []
    for sid in rows:
        rows[sid] = None                                    # would fail loudly if reached
    assert main(["run", "-c", str(config)]) == 2
    assert "state must be one of" in capsys.readouterr().err
    assert not (cfg.paths.local_dir / "subjects.csv").exists() and not called


def test_external_icv_replaces_the_registration_factor(project, tmp_path):
    config, cfg, _ = project
    ext = tmp_path / "icv.csv"
    pd.DataFrame({"subject_id": IDS, "icv_ml": np.linspace(1200, 1700, len(IDS))}).to_csv(ext, index=False)
    config.write_text("stats: {bootstrap_resamples: 200}\n"
                      f"icv: {{method: external_csv, external_csv: '{ext.as_posix()}'}}\n")
    main(["run", "-c", str(config)])
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv").set_index("subject_id")
    assert subjects.loc[IDS[0], "etiv_ml"] == pytest.approx(1200)
    assert subjects.loc[IDS[0], "head_scale"] > 1 > subjects.loc[IDS[-1], "head_scale"]
    assert (subjects["icv_method"] == "external_csv").all()


def test_add_normalized_directly():
    cfg = from_dict({})
    rng = np.random.default_rng(1)
    table = pd.DataFrame([imaging_row("a", rng), imaging_row("b", rng, reg_ok=False)])
    out = pipeline.add_normalized(table, cfg)
    assert out.loc[0, "tract_SLF_norm_ml"] == pytest.approx(table.loc[0, "tract_SLF_ml"] * table.loc[0, "head_scale"])
    assert np.isnan(out.loc[1, "tract_SLF_norm_ml"]) and np.isnan(out.loc[1, "lesion_volume_norm_ml"])
    assert out.loc[1, "lesion_count"] == table.loc[1, "lesion_count"]


def test_numbers_in_the_summary_are_written_out_not_in_exponent_form():
    from mslb.report import _fmt, _p

    assert _fmt(1182.66) == "1183" and _fmt(12345.6) == "12346"
    assert _fmt(0.5934) == "0.593" and _fmt(17.809) == "17.8" and _fmt(7) == "7"
    assert _fmt(float("nan")) == "" and _fmt(None) == ""
    assert _p(0.0004) == "<0.001" and _p(0.0164) == "0.016" and _p(float("nan")) == ""


def test_a_summary_built_on_fake_tables_says_so_and_a_real_one_does_not(project):
    config, cfg, _ = project
    main(["run", "-c", str(config)])
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert md.splitlines()[2].startswith("**FAKE DATA")
    assert json.loads((cfg.paths.shareable_dir / "group_summary.json").read_text())["fake_data"] is True
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str}).drop(columns=["is_fake"])
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    main(["run", "-c", str(config)])
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "FAKE DATA" not in md
    assert json.loads((cfg.paths.shareable_dir / "group_summary.json").read_text())["fake_data"] is False
