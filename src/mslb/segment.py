"""Lesion segmentation with MindGlide, and the lesion numbers that come out of it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mslb.config import Config
from mslb.errors import PipelineError

SEG_FILE = "seg.nii.gz"          # every MindGlide label, on the T1's grid
LESION_FILE = "lesion.nii.gz"    # the lesion mask after the size filter
SEGMENT_JSON = "segment.json"

# 26-connectivity: voxels touching by a face, an edge or a corner are one lesion
_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


class SegmentationError(PipelineError):
    """MindGlide could not segment a scan."""


def subject_dir(cfg: Config, sid: str) -> Path:
    """Where one subject's derived files live. Under outputs/local, so never shared."""
    return cfg.paths.local_dir / "subjects" / sid


def clean_lesions(mask: np.ndarray, voxel_mm3: float, min_mm3: float) -> tuple[np.ndarray, int]:
    """Drop connected components smaller than min_mm3. Returns (mask, lesion count)."""
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=_STRUCTURE)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool), 0
    sizes = np.bincount(labels.ravel(), minlength=n + 1) * voxel_mm3
    keep = sizes >= min_mm3
    keep[0] = False
    return keep[labels], int(keep.sum())


def lesion_metrics(seg: np.ndarray, voxel_mm3: float, lesion_label: int,
                   min_mm3: float) -> tuple[dict, np.ndarray]:
    """Lesion volume and count from a label image. Returns (numbers, cleaned mask).

    Volume and count describe the same mask: components under the size threshold
    are left out of both. The unfiltered volume is kept alongside for reference.
    """
    raw = seg == lesion_label
    mask, count = clean_lesions(raw, voxel_mm3, min_mm3)
    numbers = {
        "lesion_volume_ml": float(mask.sum() * voxel_mm3 / 1000),
        "lesion_count": count,
        "lesion_volume_unfiltered_ml": float(raw.sum() * voxel_mm3 / 1000),
        # every labelled voxel, cerebrospinal fluid included. a rough intracranial
        # volume, kept as a sanity check on the head-size factor, not used in tests
        "labelled_volume_ml": float((seg > 0).sum() * voxel_mm3 / 1000),
    }
    return numbers, mask


def run_mindglide(t1: Path, out: Path, device: str = "auto", sw_batch_size: int = 4) -> None:
    """Segment one scan. On a memory error, retry with the smallest batch, then on CPU."""
    import mindglide

    out.parent.mkdir(parents=True, exist_ok=True)
    attempts = [(device, sw_batch_size)]
    if sw_batch_size > 1:
        attempts.append((device, 1))
    if device != "cpu":
        attempts.append(("cpu", 1))
    last = None
    for dev, batch in attempts:
        try:
            mindglide.segment(str(t1), str(out), device=dev, sw_batch_size=batch)
            return
        except mindglide.UsageError as exc:
            raise SegmentationError(f"{t1}: {exc}") from exc
        except RuntimeError as exc:
            last = exc
            if "memory" not in str(exc).lower():
                break
            _free_gpu()
    raise SegmentationError(f"{t1}: MindGlide failed ({last})")


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def segment_subject(sid: str, t1: Path, cfg: Config, force: bool = False) -> dict:
    """Run MindGlide if needed, then compute the lesion numbers. Returns them.

    The model run is cached per scan: it is skipped when the label image is already
    there and newer than the T1. The numbers are always recomputed, since they are
    cheap and depend on settings (label, size threshold) that can change.
    """
    import nibabel as nib

    folder = subject_dir(cfg, sid)
    seg_path = folder / SEG_FILE
    fresh = seg_path.is_file() and seg_path.stat().st_mtime >= Path(t1).stat().st_mtime
    ran = False
    if force or not fresh:
        run_mindglide(Path(t1), seg_path, cfg.segmentation.device, cfg.segmentation.sw_batch_size)
        ran = True
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj).astype(np.int16)
    voxel_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
    numbers, mask = lesion_metrics(seg, voxel_mm3, cfg.segmentation.lesion_label,
                                   cfg.segmentation.min_lesion_mm3)
    lesion = nib.Nifti1Image(mask.astype(np.uint8), img.affine)
    lesion.set_data_dtype(np.uint8)
    nib.save(lesion, str(folder / LESION_FILE))
    numbers.update({"subject_id": sid, "voxel_volume_mm3": voxel_mm3, "segmented_now": ran})
    (folder / SEGMENT_JSON).write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    return numbers
