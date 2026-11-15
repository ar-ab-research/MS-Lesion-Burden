import numpy as np
import pytest
from scipy.stats import spearmanr

from mslb import fakedata
from mslb.cli import main
from mslb.config import from_dict
from mslb.inputs import InputError, discover_t1, match_ids, subject_id
from mslb.schemas import fc_column, fc_wide, load_clinical, load_fc


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_flat_and_folder_layouts(tmp_path):
    touch(tmp_path / "sub-01_T1w.nii.gz")
    touch(tmp_path / "sub-02.nii")
    touch(tmp_path / "patient03" / "T1W.nii.gz")
    touch(tmp_path / "patient03" / "T1WKS.nii.gz")        # contrast-enhanced, not the T1
    touch(tmp_path / "patient03" / "FLAIR.nii.gz")
    touch(tmp_path / "patient03" / "consensus_gt.nii.gz")
    touch(tmp_path / "sub-04" / "anat.nii.gz")            # the only image in its folder
    touch(tmp_path / "notes.txt")
    touch(tmp_path / ".hidden" / "T1w.nii.gz")
    (tmp_path / "empty").mkdir()
    found = discover_t1(tmp_path)
    assert list(found) == ["patient03", "sub-01", "sub-02", "sub-04"]
    assert found["patient03"].name == "T1W.nii.gz"


def test_ambiguous_folder_and_duplicate_subject_are_errors(tmp_path):
    touch(tmp_path / "s1" / "a.nii.gz")
    touch(tmp_path / "s1" / "b.nii.gz")
    with pytest.raises(InputError, match="cannot tell which image"):
        discover_t1(tmp_path)
    other = tmp_path / "dupes"
    touch(other / "s2.nii.gz")
    touch(other / "s2_T1w.nii.gz")
    with pytest.raises(InputError, match="two scans for subject s2"):
        discover_t1(other)
    with pytest.raises(InputError, match="not a folder"):
        discover_t1(tmp_path / "missing")


def test_subject_id_strips_only_the_t1_tag():
    assert subject_id("sub-01_T1w") == "sub-01"
    assert subject_id("sub-01-t1") == "sub-01"
    assert subject_id("visit1") == "visit1"


def test_match_ids():
    m = match_ids({"a", "b", "c"}, {"a", "b", "x"}, {"a", "c"})
    assert m["complete"] == ["a"]
    assert m["scan_without_clinical"] == ["c"]
    assert m["scan_without_fc"] == ["b"]
    assert m["clinical_without_scan"] == ["x"]


def test_fake_data_fits_the_schemas(tmp_path):
    clinical, fc = fakedata.make_fake(fakedata.default_ids(24), seed=1)
    clinical.to_csv(tmp_path / "c.csv", index=False)
    fc.to_csv(tmp_path / "f.csv", index=False)
    q5 = from_dict({"analyses": {"q5_treatment_response": True},
                    "treatment_response": {"min_nbss_improvement": 5}})
    loaded = load_clinical(tmp_path / "c.csv", q5)
    assert len(loaded) == 24 and set(loaded["do_present"]) == {0, 1}
    assert set(loaded["arm"]) == {"active", "sham"}
    wide = fc_wide(load_fc(tmp_path / "f.csv"))
    assert len(wide) == 24
    assert len([c for c in wide if c.startswith("fc__change__")]) == len(fakedata.connection_names())


def test_fake_data_is_seeded():
    a = fakedata.make_fake(fakedata.default_ids(10), seed=3)
    b = fakedata.make_fake(fakedata.default_ids(10), seed=3)
    c = fakedata.make_fake(fakedata.default_ids(10), seed=4)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not a[1].equals(c[1])


def test_planted_effects_follow_the_supplied_burden_and_nothing_else_does():
    ids = fakedata.default_ids(40)
    rng = np.random.default_rng(0)
    burden = dict(zip(ids, rng.lognormal(1, 1, 40)))
    burden[ids[0]] = burden[ids[1]] = burden[ids[2]] = 0.0     # tied zero-lesion subjects
    clinical, fc = fakedata.make_fake(ids, seed=0, burden=burden)
    b = [burden[s] for s in ids]
    wide = fc_wide(fc).set_index("subject_id").loc[ids]
    assert spearmanr(b, clinical["nbss_total"]).statistic > 0.5
    planted = fc_column("full", fakedata.PLANTED_CONNECTION)
    assert spearmanr(b, wide[planted]).statistic < -0.5
    others = [c for c in wide if c.startswith("fc__full__") and c != planted]
    assert max(abs(spearmanr(b, wide[c]).statistic) for c in others) < 0.5


def test_fake_data_needs_distinct_ids():
    with pytest.raises(ValueError):
        fakedata.make_fake(["a", "a"])


def test_cli_writes_fake_tables_and_will_not_overwrite(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    assert main(["make-fake-data", "-c", str(config), "--n", "8"]) == 0
    clinical = tmp_path / "data" / "clinical.csv"
    assert len(clinical.read_text().splitlines()) == 9
    before = clinical.read_text()
    assert main(["make-fake-data", "-c", str(config), "--seed", "5"]) == 2
    assert "not overwriting" in capsys.readouterr().err
    assert clinical.read_text() == before
    assert main(["make-fake-data", "-c", str(config), "--seed", "5", "--force"]) == 0
    assert clinical.read_text() != before


def test_cli_uses_scan_ids_when_scans_exist(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    for name in ("p1", "p2", "p3"):
        touch(tmp_path / "data" / "t1" / name / "T1w.nii.gz")
    assert main(["make-fake-data", "-c", str(config)]) == 0
    text = (tmp_path / "data" / "clinical.csv").read_text()
    assert [line.split(",")[0] for line in text.splitlines()[1:]] == ["p1", "p2", "p3"]
