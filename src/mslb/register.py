"""Registration of each T1 to the template, with ANTs.

One registration serves two purposes: its affine part gives the head-size factor,
and the full transform brings the tract atlas into the subject's own space.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mslb import atlas, icv, segment
from mslb.config import Config
from mslb.errors import PipelineError

REG_DIR = "reg"
PREFIX = "t1_to_template_"
REGISTER_JSON = "register.json"
AFFINE = PREFIX + "0GenericAffine.mat"
WARP = PREFIX + "1Warp.nii.gz"
INVERSE_WARP = PREFIX + "1InverseWarp.nii.gz"

TRANSFORMS = {"fast": "antsRegistrationSyNQuick[s]", "accurate": "antsRegistrationSyN[s]"}

# correlation between the warped T1 and the template inside the brain. good
# registrations of a T1 to a T1 template sit well above this
MIN_SIMILARITY = 0.5


class RegistrationError(PipelineError):
    """ANTs failed, or produced a transform that cannot be right."""


def reg_dir(cfg: Config, sid: str) -> Path:
    return segment.subject_dir(cfg, sid) / REG_DIR


def _settings(cfg: Config) -> dict:
    r = cfg.registration
    return {"template": r.template, "speed": r.speed, "mask_lesions": r.mask_lesions}


def _cached(folder: Path, t1: Path, cfg: Config) -> dict | None:
    record = folder / REGISTER_JSON
    files = [folder / AFFINE, folder / WARP, folder / INVERSE_WARP, record]
    if not all(f.is_file() for f in files):
        return None
    if record.stat().st_mtime < Path(t1).stat().st_mtime:
        return None
    saved = json.loads(record.read_text(encoding="utf-8"))
    return saved if saved.get("settings") == _settings(cfg) else None


def similarity(warped: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    a, b = warped[mask], template[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def register_subject(sid: str, t1: Path, cfg: Config, force: bool = False) -> dict:
    """Register one T1 to the template. Returns the head-size factor and a QC number."""
    folder = reg_dir(cfg, sid)
    if not force:
        saved = _cached(folder, t1, cfg)
        if saved is not None:
            saved["registered_now"] = False
            return saved

    import ants

    tpl = atlas.template_paths(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(tpl["t1"]))
    moving = ants.image_read(str(t1))
    kwargs = {}
    lesion_path = segment.subject_dir(cfg, sid) / segment.LESION_FILE
    if cfg.registration.mask_lesions and lesion_path.is_file():
        lesion = ants.image_read(str(lesion_path))
        if lesion.sum() > 0:
            # 1 everywhere except inside lesions, so they do not pull on the warp.
            # ants applies it to the deformable stage only, the affine sees the whole head
            kwargs["moving_mask"] = ants.threshold_image(lesion, 0.5, 1e9, 0, 1)
    try:
        reg = ants.registration(
            fixed=fixed, moving=moving, outprefix=str(folder / PREFIX),
            type_of_transform=TRANSFORMS[cfg.registration.speed], **kwargs,
        )
    except Exception as exc:  # noqa: BLE001  ants raises plain exceptions of several types
        raise RegistrationError(f"{sid}: registration failed ({exc})") from exc
    if not all((folder / f).is_file() for f in (AFFINE, WARP, INVERSE_WARP)):
        raise RegistrationError(f"{sid}: registration did not write its transforms")

    mask = ants.image_read(str(tpl["mask"])).numpy() > 0
    result = {
        "subject_id": sid,
        "head_scale": icv.head_scale(folder / AFFINE),
        "reg_similarity": similarity(reg["warpedmovout"].numpy(), fixed.numpy(), mask),
        "settings": _settings(cfg),
    }
    result["reg_ok"] = bool(icv.plausible(result["head_scale"])
                            and result["reg_similarity"] >= MIN_SIMILARITY)
    (folder / REGISTER_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["registered_now"] = True
    return result


def to_native(sid: str, t1: Path, moving_path: Path, out_path: Path, cfg: Config,
              interpolator: str = "genericLabel") -> Path:
    """Bring a template-space image onto a subject's T1 grid."""
    import ants

    folder = reg_dir(cfg, sid)
    warped = ants.apply_transforms(
        fixed=ants.image_read(str(t1)), moving=ants.image_read(str(moving_path)),
        transformlist=[str(folder / AFFINE), str(folder / INVERSE_WARP)],
        whichtoinvert=[True, False], interpolator=interpolator,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(out_path))
    return out_path
