"""Write what the slice viewer shows: small 8-bit volumes and the numbers for each scan.

Everything written here is derived from a subject's scan. By default it goes under
outputs/local, with the other per-subject files, and is served to the browser from
there. Only the public demo scans are ever exported into viewer/public/data, the
folder the public site is built from.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from mslb import segment, tracts
from mslb.config import Config
from mslb.errors import PipelineError

VOXEL_MM = 1.0          # the viewer's grid: isotropic, RAS
MARGIN_VOXELS = 4       # kept around the head when cropping
MASK_CUT = 0.4          # where a resampled mask is cut back to on or off
MAX_LESIONS = 300       # lesions listed for jump-to navigation, largest first
INDEX_FILE = "index.json"
SUMMARY_FILE = "summary.json"


def default_out(cfg: Config) -> Path:
    """Beside the other per-subject outputs, which never leave the machine."""
    return cfg.paths.local_dir / "viewer"


def _resample(img, order: int):
    """Onto a 1 mm grid whose axes run left-right, back-front, bottom-top."""
    from nibabel.processing import resample_to_output

    return resample_to_output(img, voxel_sizes=VOXEL_MM, order=order, mode="constant", cval=0)


def _head_box(t1: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of everything brighter than a tenth of the bright tissue."""
    bright = np.percentile(t1[t1 > 0], 99) if (t1 > 0).any() else 1.0
    mask = t1 > 0.1 * bright
    if not mask.any():
        return tuple(slice(0, s) for s in t1.shape)
    box = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        box.append(slice(max(hit[0] - MARGIN_VOXELS, 0),
                         min(hit[-1] + 1 + MARGIN_VOXELS, t1.shape[axis])))
    return tuple(box)


def to_uint8(t1: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale intensities to 0..255 between the 0.5th and 99.5th percentile of the head."""
    inside = t1[t1 > 0]
    lo, hi = (np.percentile(inside, [0.5, 99.5]) if inside.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((t1 - lo) / (hi - lo), 0, 1)
    return np.round(scaled * 255).astype(np.uint8), (float(lo), float(hi))


def _write(path: Path, volume: np.ndarray) -> None:
    # x fastest, then y, then z: the order a 3D texture is uploaded in
    with gzip.open(path, "wb", compresslevel=6) as out:
        out.write(np.ascontiguousarray(volume.transpose(2, 1, 0)).tobytes())


def lesion_list(mask: np.ndarray) -> list[dict]:
    """Where to jump to for each lesion, and its volume, on the viewer grid, largest first.

    The point is the lesion's own voxel nearest its centre of mass. The centre of
    mass itself can fall outside a curved lesion, for example one wrapped around a ventricle.
    """
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    boxes = ndimage.find_objects(labels)
    out = []
    for i in np.argsort(-sizes, kind="stable")[:MAX_LESIONS]:
        box = boxes[i]
        inside = np.argwhere(labels[box] == i + 1)
        nearest = inside[np.argmin(((inside - inside.mean(axis=0)) ** 2).sum(axis=1))]
        out.append({"voxel": [int(v + s.start) for v, s in zip(nearest, box)],
                    "ml": round(float(sizes[i]) * VOXEL_MM ** 3 / 1000, 4)})
    return out


def export_subject(sid: str, t1_path: Path, cfg: Config, out_dir: Path,
                   imaging_row: dict | None = None, attribution: str = "") -> Path:
    """Write one subject's viewer assets. Returns the subject's folder."""
    import nibabel as nib

    sub = segment.subject_dir(cfg, sid)
    lesion_path = sub / segment.LESION_FILE
    if not lesion_path.is_file():
        raise PipelineError(f"{sid}: no lesion mask yet, run the pipeline first")

    t1 = np.asarray(_resample(nib.load(str(t1_path)), order=1).dataobj, dtype=np.float32)
    box = _head_box(t1)
    t1_u8, window = to_uint8(t1[box])

    def layer(path: Path, order: int, threshold: float | None = None) -> np.ndarray:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)
        src = nib.Nifti1Image((data > 0).astype(np.float32) if threshold is not None
                              else data.astype(np.int16), img.affine)
        out = np.asarray(_resample(src, order=order).dataobj)[box]
        return (out > threshold).astype(np.uint8) if threshold is not None else out.astype(np.uint8)

    folder = Path(out_dir) / sid
    folder.mkdir(parents=True, exist_ok=True)
    layers = {"t1": "t1.u8.gz", "lesion": "lesion.u8.gz"}
    _write(folder / layers["t1"], t1_u8)
    # a mask is resampled as a soft image and cut just under a half: that keeps its size
    # to within a voxel at each edge without letting small lesions vanish
    lesion = layer(lesion_path, order=1, threshold=MASK_CUT)
    _write(folder / layers["lesion"], lesion)

    tract_path = sub / tracts.TRACTS_FILE
    tract_legend = {}
    if tract_path.is_file():
        from mslb import atlas

        names = atlas.read_tract_names(atlas.jhu_paths(cfg)["xml"])
        values = atlas.tract_values(cfg.tracts.primary, names)
        native = layer(tract_path, order=0)
        coded = np.zeros(native.shape, np.uint8)
        for code, (name, vals) in enumerate(values.items(), start=1):
            coded[np.isin(native, vals)] = code
            tract_legend[str(code)] = name
        layers["tracts"] = "tracts.u8.gz"
        _write(folder / layers["tracts"], coded)

    reference = Path(t1_path).parent / "consensus_gt.nii.gz"
    if reference.is_file():
        layers["reference"] = "reference.u8.gz"
        _write(folder / layers["reference"], layer(reference, order=1, threshold=MASK_CUT))

    row = imaging_row or {}

    def number(key: str):
        value = row.get(key)
        return None if value is None or not np.isfinite(value) else round(float(value), 4)

    meta = {
        "id": sid,
        "dims": [int(s) for s in t1_u8.shape],
        "voxel_mm": VOXEL_MM,
        "orientation": "RAS",
        "layers": layers,
        "tracts": tract_legend,
        "intensity_window": window,
        "lesions": lesion_list(lesion.astype(bool)),
        "stats": {
            "lesion_volume_ml": number("lesion_volume_ml"),
            "lesion_volume_norm_ml": number("lesion_volume_norm_ml"),
            "lesion_count": None if row.get("lesion_count") is None else int(row["lesion_count"]),
            "head_scale": number("head_scale"),
            "tracts": {name: {"ml": number(f"tract_{name}_ml"),
                              "norm_ml": number(f"tract_{name}_norm_ml"),
                              "pct": number(f"tract_{name}_pct")}
                       for name in cfg.tracts.primary},
        },
        "attribution": attribution,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder


def write_index(out_dir: Path) -> list[str]:
    """List every exported scan with its measurements, for the menu and the per-scan table."""
    out_dir = Path(out_dir)
    metas = sorted(out_dir.glob("*/meta.json"), key=lambda p: p.parent.name)
    rows = []
    for path in metas:
        meta = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"id": meta["id"], **meta["stats"]})
    index = {"subjects": [row["id"] for row in rows], "measurements": rows}
    (out_dir / INDEX_FILE).write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index["subjects"]


def write_summary(cfg: Config, out_dir: Path) -> bool:
    """Put the group summary where the viewer can read it. False when there is none yet."""
    source = cfg.paths.shareable_dir / "group_summary.json"
    target = Path(out_dir) / SUMMARY_FILE
    if not source.is_file():
        target.unlink(missing_ok=True)
        return False
    summary = json.loads(source.read_text(encoding="utf-8"))
    text = cfg.paths.shareable_dir / "group_summary.md"
    summary["markdown"] = text.read_text(encoding="utf-8") if text.is_file() else ""
    target.write_text(json.dumps(summary), encoding="utf-8")
    return True


def _up_to_date(folder: Path, sources: list[Path]) -> bool:
    meta = folder / "meta.json"
    if not meta.is_file():
        return False
    return all(meta.stat().st_mtime >= s.stat().st_mtime for s in sources if s.is_file())


def export(cfg: Config, scans: dict[str, Path], out_dir: Path, subjects=None,
           attribution: str = "", force: bool = False, log=print) -> list[str]:
    """Export the named subjects (default: all that have been processed).

    A subject whose files are newer than its lesion mask and the subject table is left
    as it is. Returns the IDs the viewer will list.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    rows = {}
    table_path = cfg.paths.local_dir / "subjects.csv"
    if table_path.is_file():
        table = pd.read_csv(table_path, dtype={"subject_id": str})
        rows = {r["subject_id"]: r for r in table.to_dict("records")}
    wanted = list(subjects) if subjects else [
        s for s in scans if (segment.subject_dir(cfg, s) / segment.LESION_FILE).is_file()]
    unknown = [s for s in wanted if s not in scans]
    if unknown:
        raise PipelineError(f"no scan for: {', '.join(unknown)}")
    for sid in wanted:
        lesion = segment.subject_dir(cfg, sid) / segment.LESION_FILE
        if not force and _up_to_date(out_dir / sid, [lesion, table_path]):
            continue
        export_subject(sid, scans[sid], cfg, out_dir, rows.get(sid), attribution)
        log(f"viewer    {sid}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(cfg, out_dir)
    return write_index(out_dir)
