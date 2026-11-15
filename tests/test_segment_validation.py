import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mslb import segment, validation
from mslb.config import from_dict
from mslb.segment import clean_lesions, lesion_metrics
from mslb.validation import bland_altman, lesion_detection, overlap, summarise


def blank(shape=(20, 20, 20)):
    return np.zeros(shape, dtype=np.int16)


def test_volume_and_count_on_a_known_image():
    seg = blank()
    seg[2:5, 2:5, 2:5] = 18          # 27 voxels
    seg[10:12, 10:12, 10:12] = 18    # 8 voxels
    seg[15, 15, 15] = 18             # 1 voxel, under the threshold
    seg[0:2, 0:2, 0:2] = 13          # white matter, not a lesion
    numbers, mask = lesion_metrics(seg, voxel_mm3=1.0, lesion_label=18, min_mm3=3.0)
    assert numbers["lesion_count"] == 2
    assert numbers["lesion_volume_ml"] == pytest.approx(0.035)
    assert numbers["lesion_volume_unfiltered_ml"] == pytest.approx(0.036)
    assert numbers["labelled_volume_ml"] == pytest.approx(0.044)
    assert mask.sum() == 35 and not mask[15, 15, 15]


def test_voxel_size_scales_volume_and_moves_the_threshold():
    seg = blank()
    seg[2:4, 2:4, 2:4] = 18                                   # 8 voxels
    small, _ = lesion_metrics(seg, voxel_mm3=0.343, lesion_label=18, min_mm3=3.0)
    assert small["lesion_count"] == 0 and small["lesion_volume_ml"] == 0     # 2.7 mm3
    big, _ = lesion_metrics(seg, voxel_mm3=0.5, lesion_label=18, min_mm3=3.0)
    assert big["lesion_count"] == 1 and big["lesion_volume_ml"] == pytest.approx(0.004)


def test_threshold_is_inclusive_and_zero_keeps_everything():
    mask = np.zeros((10, 10, 10), bool)
    mask[1, 1, 1:4] = True     # exactly 3 mm3 at 1 mm voxels
    mask[8, 8, 8] = True
    assert clean_lesions(mask, 1.0, 3.0)[1] == 1
    assert clean_lesions(mask, 1.0, 0.0)[1] == 2
    assert clean_lesions(np.zeros((4, 4, 4), bool), 1.0, 3.0) == (pytest.approx(np.zeros((4, 4, 4))), 0)


def test_corner_touching_voxels_are_one_lesion():
    mask = np.zeros((6, 6, 6), bool)
    mask[1, 1, 1] = mask[2, 2, 2] = mask[3, 3, 3] = True
    assert clean_lesions(mask, 1.0, 3.0)[1] == 1


@pytest.fixture
def subject(tmp_path, monkeypatch):
    """A fake T1 on disk and a stand-in for MindGlide that writes a known label image."""
    cfg = from_dict({}, base=tmp_path)
    affine = np.diag([0.5, 0.5, 2.0, 1.0])            # 0.5 mm3 voxels
    t1 = tmp_path / "data" / "t1" / "s1.nii.gz"
    t1.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), np.float32), affine), str(t1))
    calls = []

    def fake_mindglide(t1_path, out, device="auto", sw_batch_size=4):
        calls.append((device, sw_batch_size))
        seg = blank()
        seg[2:6, 2:6, 2:6] = 18                        # 64 voxels = 32 mm3
        out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(seg, affine), str(out))

    monkeypatch.setattr(segment, "run_mindglide", fake_mindglide)
    return cfg, t1, calls


def test_segment_subject_writes_mask_and_numbers(subject):
    cfg, t1, calls = subject
    numbers = segment.segment_subject("s1", t1, cfg)
    assert numbers["lesion_volume_ml"] == pytest.approx(0.032)
    assert numbers["voxel_volume_mm3"] == pytest.approx(0.5)
    folder = segment.subject_dir(cfg, "s1")
    assert folder == cfg.paths.local_dir / "subjects" / "s1"
    lesion = nib.load(str(folder / segment.LESION_FILE))
    assert np.asarray(lesion.dataobj).sum() == 64
    assert np.allclose(lesion.affine, nib.load(str(t1)).affine)
    assert (folder / segment.SEGMENT_JSON).is_file()
    assert calls == [("auto", 4)]


def test_model_run_is_cached_but_numbers_follow_the_settings(subject, tmp_path):
    cfg, t1, calls = subject
    segment.segment_subject("s1", t1, cfg)
    stricter = from_dict({"segmentation": {"min_lesion_mm3": 100}}, base=tmp_path)
    again = segment.segment_subject("s1", t1, stricter)
    assert len(calls) == 1                      # no second model run
    assert again["lesion_count"] == 0           # but the new threshold applied
    segment.segment_subject("s1", t1, cfg, force=True)
    assert len(calls) == 2


def test_a_newer_t1_invalidates_the_cache(subject):
    cfg, t1, calls = subject
    segment.segment_subject("s1", t1, cfg)
    seg_time = (segment.subject_dir(cfg, "s1") / segment.SEG_FILE).stat().st_mtime
    os.utime(t1, (seg_time + 10, seg_time + 10))
    segment.segment_subject("s1", t1, cfg)
    assert len(calls) == 2


def test_overlap_known_values():
    ref = np.zeros((10, 10, 10), bool)
    pred = np.zeros((10, 10, 10), bool)
    ref[0:4, 0, 0] = True          # 4 voxels
    pred[2:8, 0, 0] = True         # 6 voxels, 2 shared
    o = overlap(pred, ref)
    assert o["dice"] == pytest.approx(2 * 2 / 10)
    assert o["sensitivity"] == pytest.approx(0.5)
    assert o["precision"] == pytest.approx(2 / 6)
    assert overlap(ref, ref)["dice"] == 1.0
    assert overlap(pred, ~pred)["dice"] == 0.0


def test_overlap_with_empty_masks_is_missing_not_perfect():
    empty = np.zeros((5, 5, 5), bool)
    full = np.ones((5, 5, 5), bool)
    assert np.isnan(overlap(empty, empty)["dice"])
    assert overlap(empty, full)["dice"] == 0.0
    assert np.isnan(overlap(empty, full)["precision"])


def test_lesion_detection_counts_lesions_not_voxels():
    ref = np.zeros((20, 20, 20), bool)
    pred = np.zeros((20, 20, 20), bool)
    ref[1:4, 1:4, 1:4] = True       # found: one predicted voxel touches it
    ref[10:13, 10:13, 10:13] = True  # missed
    pred[3, 3, 3] = True
    pred[17, 17, 17] = True         # false positive
    d = lesion_detection(pred, ref)
    assert d["ref_lesions"] == 2 and d["pred_lesions"] == 2
    assert d["lesion_sensitivity"] == 0.5 and d["lesion_precision"] == 0.5


def test_bland_altman_and_summary():
    ba = bland_altman([2, 4, 6], [1, 2, 3])
    assert ba["bias"] == pytest.approx(2.0)
    assert ba["loa_high"] - ba["bias"] == pytest.approx(1.96 * 1.0)
    rows = pd.DataFrame({
        "pred_volume_ml": [1.0, 2.0, 3.0, 4.5], "ref_volume_ml": [2.0, 4.0, 6.0, 8.0],
        "dice": [0.2, 0.4, 0.6, np.nan], "sensitivity": [0.1, 0.2, 0.3, 0.4],
        "precision": [0.5] * 4, "lesion_sensitivity": [0.5] * 4, "lesion_precision": [0.5] * 4,
    })
    s = summarise(rows)
    assert s["n"] == 4
    assert s["dice"]["mean"] == pytest.approx(0.4)          # the missing dice is left out
    assert s["volume"]["spearman_rho"] == pytest.approx(1.0)
    assert s["volume"]["bland_altman_ml"]["bias"] < 0
    assert "Dice" in validation.report_markdown(s)


def test_compare_subject_refuses_mismatched_grids(subject, tmp_path):
    cfg, t1, _ = subject
    folder = tmp_path / "openms" / "s1"
    folder.mkdir(parents=True)
    nib.save(nib.load(str(t1)), str(folder / "T1W.nii.gz"))
    ref = np.zeros((20, 20, 20), np.uint8)
    ref[2:6, 2:6, 2:6] = 1
    nib.save(nib.Nifti1Image(ref, np.diag([0.5, 0.5, 2.0, 1.0])), str(folder / "consensus_gt.nii.gz"))
    row = validation.compare_subject("s1", folder, cfg)
    assert row["dice"] == 1.0 and row["pred_volume_ml"] == pytest.approx(0.032)
    nib.save(nib.Nifti1Image(ref, np.diag([1.0, 1.0, 1.0, 1.0])), str(folder / "consensus_gt.nii.gz"))
    with pytest.raises(Exception, match="same grid"):
        validation.compare_subject("s1", folder, cfg)
