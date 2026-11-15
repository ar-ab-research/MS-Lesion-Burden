"""Find the T1 scans and line them up with the tables."""
from __future__ import annotations

import re
from pathlib import Path

from mslb.errors import PipelineError


class InputError(PipelineError):
    """The scans folder cannot be read unambiguously, or a scan is unusable."""


_NIFTI = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
_T1_TAG = re.compile(r"(^|[_-])t1w?$", re.IGNORECASE)


def _stem(path: Path) -> str:
    return _NIFTI.sub("", path.name)


def _is_nifti(path: Path) -> bool:
    return path.is_file() and bool(_NIFTI.search(path.name))


def subject_id(stem: str) -> str:
    """sub-01_T1w and sub-01 are the same subject."""
    return re.sub(r"[_-]t1w?$", "", stem, flags=re.IGNORECASE) or stem


def discover_t1(t1_dir: Path) -> dict[str, Path]:
    """Subject ID to T1 path, sorted by ID.

    Two layouts are read, and may be mixed:
      <id>.nii.gz or <id>_T1w.nii.gz     directly in the folder
      <id>/T1w.nii.gz                     one folder per subject

    A subject folder may hold other images. The T1 is the file named T1w (or ending
    in _T1w); with no such file, a folder holding exactly one image uses that one.
    """
    t1_dir = Path(t1_dir)
    if not t1_dir.is_dir():
        raise InputError(f"not a folder: {t1_dir}")
    found: dict[str, Path] = {}

    def add(sid: str, path: Path) -> None:
        if sid in found:
            raise InputError(f"two scans for subject {sid}: {found[sid]} and {path}")
        found[sid] = path

    for entry in sorted(t1_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if _is_nifti(entry):
            add(subject_id(_stem(entry)), entry)
        elif entry.is_dir():
            images = sorted(p for p in entry.iterdir() if _is_nifti(p))
            if not images:
                continue
            tagged = [p for p in images if _T1_TAG.search(_stem(p))]
            if len(tagged) == 1:
                add(entry.name, tagged[0])
            elif not tagged and len(images) == 1:
                add(entry.name, images[0])
            else:
                names = ", ".join(p.name for p in (tagged or images))
                raise InputError(
                    f"cannot tell which image in {entry} is the T1 ({names}). "
                    "name it T1w.nii.gz"
                )
    return dict(sorted(found.items()))


def check_t1(path: Path) -> dict:
    """Header checks. Returns shape and voxel size, or raises InputError."""
    import nibabel as nib
    import numpy as np

    try:
        img = nib.load(str(path))
    except Exception as exc:  # noqa: BLE001  nibabel raises several unrelated types
        raise InputError(f"{path}: not a readable NIfTI ({exc})") from exc
    shape = tuple(int(s) for s in img.shape if s != 1) if img.ndim > 3 else tuple(img.shape)
    if len(shape) != 3:
        raise InputError(f"{path}: expected a 3D volume, shape is {tuple(img.shape)}")
    affine = np.asarray(img.affine, dtype=float)
    det = float(np.linalg.det(affine[:3, :3]))
    if not np.isfinite(affine).all() or abs(det) < 1e-6:
        raise InputError(f"{path}: the affine is not usable")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if min(zooms) <= 0 or max(zooms) > 10:
        raise InputError(f"{path}: voxel size {zooms} mm is not plausible")
    return {"shape": shape, "voxel_mm": zooms, "voxel_volume_mm3": abs(det)}


def match_ids(scans: set[str], clinical: set[str], fc: set[str]) -> dict[str, list[str]]:
    """Who is where. Every scan is processed; the tables decide who enters which test."""
    return {
        "scan_without_clinical": sorted(scans - clinical),
        "scan_without_fc": sorted(scans - fc),
        "clinical_without_scan": sorted(clinical - scans),
        "fc_without_scan": sorted(fc - scans),
        "complete": sorted(scans & clinical & fc),
    }
