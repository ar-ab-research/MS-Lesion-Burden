import gzip
import json

import nibabel as nib
import numpy as np
import pytest

from mslb import segment, viewer_export
from mslb.config import from_dict
from mslb.errors import PipelineError


@pytest.fixture
def exported(tmp_path):
    """A head-shaped T1 at 2 mm with a left-right flipped header, and a lesion in it."""
    cfg = from_dict({}, base=tmp_path)
    shape = (40, 48, 36)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])       # x index runs right to left
    affine[:3, 3] = [40, -48, -36]
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    centre = (np.array(shape) - 1) / 2
    head = (((grid - centre) / (np.array(shape) * 0.35)) ** 2).sum(-1) <= 1
    t1 = np.where(head, 600.0, 0.0).astype(np.float32)
    lesion = np.zeros(shape, np.uint8)
    lesion[10:13, 20:23, 18:21] = 1                # 27 voxels of 8 mm3 = 0.216 mL
    t1_path = tmp_path / "scans" / "s1" / "T1w.nii.gz"
    t1_path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(t1, affine), str(t1_path))
    sub = segment.subject_dir(cfg, "s1")
    sub.mkdir(parents=True)
    nib.save(nib.Nifti1Image(lesion, affine), str(sub / segment.LESION_FILE))
    out = tmp_path / "viewer_data"
    row = {"lesion_volume_ml": 0.216, "lesion_volume_norm_ml": 0.3, "lesion_count": 1,
           "head_scale": 1.39, "tract_ATR_ml": 0.0, "tract_ATR_norm_ml": 0.0, "tract_ATR_pct": 0.0,
           "tract_SLF_ml": float("nan"), "tract_SLF_pct": float("nan")}
    folder = viewer_export.export_subject("s1", t1_path, cfg, out, row, "credit line")
    return cfg, t1_path, out, folder


def read(folder, name, dims):
    data = np.frombuffer(gzip.decompress((folder / name).read_bytes()), np.uint8)
    # written x fastest, so it reads back as z, y, x
    return data.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)


def test_volumes_are_one_mm_cropped_to_the_head_and_the_right_size(exported):
    _, _, _, folder = exported
    meta = json.loads((folder / "meta.json").read_text())
    dims = meta["dims"]
    assert meta["voxel_mm"] == 1.0 and meta["orientation"] == "RAS"
    # the head is about 56 x 67 x 50 mm across. cropped to it, plus a small margin
    for got, want in zip(dims, (56, 67, 50)):
        assert want <= got <= want + 2 * viewer_export.MARGIN_VOXELS + 4
    t1 = read(folder, meta["layers"]["t1"], dims)
    assert t1.max() == 255 and t1.min() == 0
    lesion = read(folder, meta["layers"]["lesion"], dims)
    assert set(np.unique(lesion)) == {0, 1}
    # a 6 mm cube. resampling may move each face by up to one voxel, no more
    assert 5 ** 3 <= lesion.sum() <= 7 ** 3


def test_the_lesion_ends_up_on_the_correct_side(exported):
    _, _, _, folder = exported
    meta = json.loads((folder / "meta.json").read_text())
    lesion = read(folder, meta["layers"]["lesion"], meta["dims"])
    x = np.argwhere(lesion)[:, 0].mean()
    # stored x index 10 to 12 of 40 with a negative x step is on the subject's right,
    # which is the high end of the viewer's left-to-right axis
    assert x > meta["dims"][0] / 2
    assert meta["lesions"][0]["voxel"][0] == pytest.approx(x, abs=1.5)


def test_meta_carries_the_numbers_and_leaves_missing_ones_empty(exported):
    _, _, _, folder = exported
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["stats"]["lesion_volume_ml"] == 0.216 and meta["stats"]["lesion_count"] == 1
    assert meta["stats"]["tracts"]["ATR"] == {"ml": 0.0, "norm_ml": 0.0, "pct": 0.0}
    assert meta["stats"]["tracts"]["SLF"] == {"ml": None, "norm_ml": None, "pct": None}
    assert meta["attribution"] == "credit line"
    assert len(meta["lesions"]) == 1 and 0.125 <= meta["lesions"][0]["ml"] <= 0.343
    assert set(meta["layers"]) == {"t1", "lesion"}                 # no tracts, no reference here


def test_a_reference_mask_beside_the_scan_becomes_a_layer(exported, tmp_path):
    cfg, t1_path, out, _ = exported
    img = nib.load(str(t1_path))
    ref = np.zeros(img.shape, np.uint8)
    ref[20:24, 20:24, 10:14] = 1
    nib.save(nib.Nifti1Image(ref, img.affine), str(t1_path.parent / "consensus_gt.nii.gz"))
    folder = viewer_export.export_subject("s1", t1_path, cfg, out)
    meta = json.loads((folder / "meta.json").read_text())
    assert "reference" in meta["layers"]
    assert read(folder, meta["layers"]["reference"], meta["dims"]).sum() > 0
    assert meta["stats"]["lesion_volume_ml"] is None                # no table row given


def test_index_lists_what_is_there_and_unprocessed_subjects_are_refused(exported):
    cfg, t1_path, out, _ = exported
    assert viewer_export.write_index(out) == ["s1"]
    index = json.loads((out / "index.json").read_text())
    assert index["subjects"] == ["s1"]
    # the per-scan table in the viewer is drawn from these rows
    assert index["measurements"][0]["id"] == "s1"
    assert index["measurements"][0]["lesion_volume_ml"] == 0.216
    with pytest.raises(PipelineError, match="no lesion mask yet"):
        viewer_export.export_subject("s2", t1_path, cfg, out)
    with pytest.raises(PipelineError, match="no scan for: ghost"):
        viewer_export.export(cfg, {"s1": t1_path}, out, subjects=["ghost"])
    assert viewer_export.export(cfg, {"s1": t1_path}, out, log=lambda *_: None) == ["s1"]


def test_the_jump_point_of_a_curved_lesion_is_inside_it():
    mask = np.zeros((30, 30, 30), bool)
    mask[5:25, 5:25, 14:16] = True
    mask[8:22, 8:25, 14:16] = False          # a C shape: its centre of mass is in the gap
    mask[2, 2, 2] = True                     # and a one-voxel lesion
    lesions = viewer_export.lesion_list(mask)
    assert [round(l["ml"], 3) for l in lesions] == [round(mask.sum() / 1000 - 0.001, 3), 0.001]
    x, y, z = lesions[0]["voxel"]
    assert mask[x, y, z]
    assert lesions[1]["voxel"] == [2, 2, 2]
    assert viewer_export.lesion_list(np.zeros((4, 4, 4), bool)) == []


def test_export_skips_what_is_up_to_date_and_copies_the_group_summary(exported):
    cfg, t1_path, out, folder = exported
    logged = []
    viewer_export.export(cfg, {"s1": t1_path}, out, log=logged.append)
    assert logged == []                                   # nothing changed, nothing redone
    assert not (out / "summary.json").exists()            # and no summary before a run
    viewer_export.export(cfg, {"s1": t1_path}, out, force=True, log=logged.append)
    assert logged == ["viewer    s1"]

    share = cfg.paths.shareable_dir
    share.mkdir(parents=True)
    (share / "group_summary.json").write_text(json.dumps({"fake_data": True, "tests": []}))
    (share / "group_summary.md").write_text("# Group summary")
    viewer_export.export(cfg, {"s1": t1_path}, out, log=logged.append)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fake_data"] is True and summary["markdown"] == "# Group summary"
    (share / "group_summary.json").unlink()
    viewer_export.export(cfg, {"s1": t1_path}, out, log=logged.append)
    assert not (out / "summary.json").exists()            # a stale summary does not linger
