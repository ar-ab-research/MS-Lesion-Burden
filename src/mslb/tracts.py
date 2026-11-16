"""Lesion volume inside each white-matter tract.

The tract atlas is brought into the subject's own space, so volumes are measured
on the subject's own voxels and no lesion mask is ever resampled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mslb import atlas, register, segment
from mslb.config import Config
from mslb.errors import PipelineError

TRACTS_FILE = "tracts.nii.gz"    # atlas labels on the T1's grid
TRACTS_JSON = "tracts.json"


def tract_loads(labels: np.ndarray, lesion: np.ndarray, groups: dict[str, list[int]],
                voxel_mm3: float) -> dict:
    """For each group of atlas labels: lesion volume inside it, its own volume, and the share lesioned."""
    lesion = lesion.astype(bool)
    out = {}
    for name, values in groups.items():
        tract = np.isin(labels, values)
        tract_ml = float(tract.sum() * voxel_mm3 / 1000)
        lesion_ml = float((tract & lesion).sum() * voxel_mm3 / 1000)
        out[f"tract_{name}_ml"] = lesion_ml
        out[f"tract_{name}_size_ml"] = tract_ml
        out[f"tract_{name}_pct"] = 100 * lesion_ml / tract_ml if tract_ml else np.nan
    return out


def tract_subject(sid: str, t1: Path, cfg: Config) -> dict:
    """Warp the atlas to one subject and measure lesion load per tract."""
    import nibabel as nib

    folder = segment.subject_dir(cfg, sid)
    paths = atlas.jhu_paths(cfg)
    names = atlas.read_tract_names(paths["xml"])
    groups = atlas.tract_values({**cfg.tracts.primary, **cfg.tracts.secondary}, names)

    # the warped atlas only changes when the registration does
    native = folder / TRACTS_FILE
    affine = register.reg_dir(cfg, sid) / register.AFFINE
    if not native.is_file() or native.stat().st_mtime < affine.stat().st_mtime:
        register.to_native(sid, t1, paths["labels"], native, cfg)
    labels_img = nib.load(str(native))
    lesion_img = nib.load(str(folder / segment.LESION_FILE))
    if labels_img.shape != lesion_img.shape or not np.allclose(
            labels_img.affine, lesion_img.affine, atol=1e-3):
        raise PipelineError(f"{sid}: the warped atlas and the lesion mask are not on the same grid")
    labels = np.asarray(labels_img.dataobj).astype(np.int16)
    lesion = np.asarray(lesion_img.dataobj) > 0
    voxel_mm3 = float(abs(np.linalg.det(lesion_img.affine[:3, :3])))
    out = {"subject_id": sid}
    out.update(tract_loads(labels, lesion, groups, voxel_mm3))
    (folder / TRACTS_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
