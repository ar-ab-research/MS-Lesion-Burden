from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mslb import schemas
from mslb.config import from_dict
from mslb.schemas import SchemaError, fc_column, fc_wide, load_clinical, load_fc

ROOT = Path(__file__).resolve().parents[1]
CFG = from_dict({})
Q5 = from_dict({"analyses": {"q5_treatment_response": True},
                "treatment_response": {"min_nbss_improvement": 5}})

CLINICAL = (
    "subject_id,do_present,nbss_total,diary_voids_per_day,diary_urgency_per_day,diary_incontinence_per_day\n"
    "001,1,30,10,4,1\n"
    "002,0,22,8.5,2,0\n"
    "003,,41,,6,2\n"
)


def write(tmp_path, text, name="table.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_committed_templates_match_the_schema(tmp_path):
    # templates/ is what users copy. it is generated, and must not drift from the code
    for path in schemas.write_templates(tmp_path):
        committed = ROOT / "templates" / path.name
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), path.name


def test_templates_are_headers_only():
    for table in ("clinical", "fc"):
        frame = pd.read_csv(ROOT / "templates" / f"{table}_template.csv")
        assert len(frame) == 0
        assert list(frame.columns) == [c.name for c in schemas.columns(table)]


def test_clinical_loads_with_types_and_blanks(tmp_path):
    frame = load_clinical(write(tmp_path, CLINICAL), CFG)
    assert list(frame["subject_id"]) == ["001", "002", "003"]      # zeros kept
    assert frame["nbss_total"].tolist() == [30, 22, 41]
    assert np.isnan(frame.loc[2, "do_present"])                    # blank is missing, not an error
    assert np.isnan(frame.loc[2, "diary_voids_per_day"])


def test_q5_columns_only_required_when_q5_is_on(tmp_path):
    path = write(tmp_path, CLINICAL)
    load_clinical(path, CFG)
    with pytest.raises(SchemaError, match="arm"):
        load_clinical(path, Q5)


def test_a_column_for_a_disabled_analysis_may_be_absent(tmp_path):
    cfg = from_dict({"analyses": {"q3_detrusor_overactivity": False}})
    text = "subject_id,nbss_total,diary_voids_per_day,diary_urgency_per_day,diary_incontinence_per_day\n1,30,10,4,1\n"
    assert "do_present" not in load_clinical(write(tmp_path, text), cfg)


def test_extra_columns_are_carried_through(tmp_path):
    text = "subject_id,site,nbss_total,do_present,diary_voids_per_day,diary_urgency_per_day,diary_incontinence_per_day\n1,A,30,1,10,4,1\n"
    assert load_clinical(write(tmp_path, text), CFG).loc[0, "site"] == "A"


@pytest.mark.parametrize("bad_row, needle", [
    ("004,2,30,10,4,1", "do_present must be 0 or 1"),
    ("004,1,90,10,4,1", "nbss_total is above 74"),
    ("004,1,thirty,10,4,1", "nbss_total must be a number"),
    ("004,1,30,-1,4,1", "diary_voids_per_day is below 0"),
    (",1,30,10,4,1", "subject_id is empty"),
    ("001,1,30,10,4,1", "more than once"),
])
def test_bad_clinical_cells_are_named(tmp_path, bad_row, needle):
    with pytest.raises(SchemaError, match=needle):
        load_clinical(write(tmp_path, CLINICAL + bad_row + "\n"), CFG)


def test_every_problem_is_reported_at_once_with_file_rows(tmp_path):
    text = CLINICAL + "004,2,90,10,4,1\n"
    with pytest.raises(SchemaError) as info:
        load_clinical(write(tmp_path, text), CFG)
    message = str(info.value)
    assert "do_present" in message and "nbss_total" in message
    assert "file row 5" in message            # header is row 1


def test_missing_column_is_named(tmp_path):
    with pytest.raises(SchemaError, match="missing column.*nbss_total"):
        load_clinical(write(tmp_path, "subject_id,do_present\n1,1\n"), CFG)


def test_bad_arm_value(tmp_path):
    text = CLINICAL.splitlines()[0] + ",arm,nbss_total_post\n001,1,30,10,4,1,placebo,20\n"
    with pytest.raises(SchemaError, match="arm must be one of active"):
        load_clinical(write(tmp_path, text), Q5)


FC = (
    "subject_id,state,connection,fc_z\n"
    "001,full,a__b,0.50\n001,empty,a__b,0.20\n"
    "002,full,a__b,0.10\n002,empty,a__b,0.40\n"
    "001,full,c__d,0.30\n001,empty,c__d,0.30\n001,change,c__d,0.99\n"
    "002,full,c__d,0.60\n"
)


def test_fc_change_is_full_minus_empty(tmp_path):
    wide = fc_wide(load_fc(write(tmp_path, FC))).set_index("subject_id")
    assert wide.loc["001", fc_column("change", "a__b")] == pytest.approx(0.30)
    assert wide.loc["002", fc_column("change", "a__b")] == pytest.approx(-0.30)


def test_supplied_change_wins_and_incomplete_pairs_stay_missing(tmp_path):
    wide = fc_wide(load_fc(write(tmp_path, FC))).set_index("subject_id")
    assert wide.loc["001", fc_column("change", "c__d")] == pytest.approx(0.99)
    assert np.isnan(wide.loc["002", fc_column("change", "c__d")])   # no empty scan, no change
    assert wide.loc["002", fc_column("full", "c__d")] == pytest.approx(0.60)


@pytest.mark.parametrize("row, needle", [
    ("001,full,a__b,0.7", "repeat a subject, state and connection"),
    ("003,resting,a__b,0.7", "state must be one of"),
    ("003,full,a__b,high", "fc_z must be a number"),
    ("003,full,,0.1", "connection is empty"),
])
def test_bad_fc_rows(tmp_path, row, needle):
    with pytest.raises(SchemaError, match=needle):
        load_fc(write(tmp_path, FC + row + "\n"))


def test_state_is_case_insensitive_and_bom_is_tolerated(tmp_path):
    text = "﻿subject_id,state,connection,fc_z\n1,Full,a,0.1\n1,EMPTY,a,0.2\n"
    assert sorted(load_fc(write(tmp_path, text))["state"]) == ["empty", "full"]


def test_not_a_csv(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"\x00\x01\x02\xff\xfe")
    with pytest.raises(SchemaError):
        load_fc(path)
