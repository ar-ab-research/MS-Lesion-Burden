"""Head size, from the affine part of the registration to the template.

The idea is the one behind FreeSurfer's eTIV (Buckner et al. 2004) and SIENAX's
scaling factor: the amount a whole head has to be stretched to fit a template
tracks intracranial volume, and it can be measured on a T1 alone.

head_scale = template volume / subject volume. It is above 1 for a head smaller
than the template. A normalized volume is the raw volume times head_scale, which
is the volume the lesions would have if the head were the size of the template.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mslb.config import Config
from mslb.errors import PipelineError

# the template is larger than most heads, so real values sit around 1.1 to 1.6.
# far outside that is a failed registration, not anatomy
PLAUSIBLE = (0.7, 2.0)


class IcvError(PipelineError):
    """The head-size factor could not be worked out."""


def affine_matrix(mat_path: Path) -> np.ndarray:
    """The 3x3 linear part of an ITK/ANTs affine .mat file."""
    from scipy.io import loadmat

    try:
        content = loadmat(str(mat_path))
    except Exception as exc:  # noqa: BLE001  scipy raises several unrelated types
        raise IcvError(f"{mat_path}: not a readable transform ({exc})") from exc
    keys = [k for k in content if k.startswith(("AffineTransform", "MatrixOffsetTransformBase"))]
    if len(keys) != 1:
        raise IcvError(f"{mat_path}: expected one affine transform, found {len(keys)}")
    params = np.asarray(content[keys[0]], dtype=float).ravel()
    if params.size != 12:
        raise IcvError(f"{mat_path}: expected a 3D affine with 12 parameters, got {params.size}")
    return params[:9].reshape(3, 3)


def head_scale_from_matrix(matrix: np.ndarray) -> float:
    """Template volume over subject volume.

    ANTs stores the transform that takes a point in the fixed image (the template)
    to the matching point in the moving image (the subject). Its determinant is
    therefore subject volume over template volume, and the scale is its inverse.
    """
    det = abs(float(np.linalg.det(matrix)))
    if not np.isfinite(det) or det <= 0:
        raise IcvError("the affine transform is singular")
    return 1.0 / det


def head_scale(mat_path: Path) -> float:
    return head_scale_from_matrix(affine_matrix(mat_path))


def plausible(scale: float) -> bool:
    return PLAUSIBLE[0] <= scale <= PLAUSIBLE[1]


def mask_volume_ml(mask_path: Path) -> float:
    import nibabel as nib

    img = nib.load(str(mask_path))
    voxel = abs(float(np.linalg.det(img.affine[:3, :3])))
    return float((np.asarray(img.dataobj) > 0).sum() * voxel / 1000)


def external_scales(cfg: Config, ids: list[str]) -> pd.DataFrame:
    """Head-size numbers from a table of intracranial volumes measured elsewhere.

    The table needs subject_id and icv_ml. head_scale is the mean ICV of the
    listed subjects over each subject's own, so normalized volumes stay in mL.
    """
    path = cfg.icv.external_csv
    try:
        table = pd.read_csv(path, dtype={"subject_id": str})
    except Exception as exc:  # noqa: BLE001
        raise IcvError(f"{path}: cannot be read as csv ({exc})") from exc
    if not {"subject_id", "icv_ml"} <= set(table.columns):
        raise IcvError(f"{path}: needs the columns subject_id and icv_ml")
    table = table.assign(subject_id=table["subject_id"].str.strip()).set_index("subject_id")
    icv = pd.to_numeric(table["icv_ml"], errors="coerce").reindex(ids)
    if (icv <= 0).any():
        raise IcvError(f"{path}: icv_ml must be positive")
    out = pd.DataFrame({"subject_id": ids, "etiv_ml": icv.to_numpy()})
    out["head_scale"] = icv.mean() / icv.to_numpy()
    out["icv_method"] = "external_csv"
    return out
