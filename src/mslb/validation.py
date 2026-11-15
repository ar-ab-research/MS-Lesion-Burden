"""Agreement between MindGlide on T1 alone and expert lesion masks.

Run on the public Open MS dataset (Lesjak et al. 2017, 30 patients, CC-BY), whose
consensus masks were drawn with FLAIR and T2 available. The numbers are reported
as they come: nothing in the pipeline is tuned on this dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mslb import segment
from mslb.config import Config
from mslb.download import fetch
from mslb.errors import PipelineError

OPENMS_URL = ("https://raw.githubusercontent.com/muschellij2/open_ms_data/master/"
              "cross_sectional/coregistered")
OPENMS_FILES = ("T1W.nii.gz", "consensus_gt.nii.gz", "brainmask.nii.gz")
OPENMS_N = 30
OPENMS_CITATION = (
    "Lesjak Z, et al. A novel public MR image dataset of multiple "
    "sclerosis patients with lesion segmentations based on multi-rater consensus. "
    "Neuroinformatics 2018;16:51-63. doi:10.1007/s12021-017-9348-7. Licence: CC-BY."
)


def openms_ids() -> list[str]:
    return [f"patient{i:02d}" for i in range(1, OPENMS_N + 1)]


def fetch_openms(dest: Path, ids=None, log=print) -> Path:
    """Download the co-registered T1, consensus mask and brain mask per patient."""
    dest = Path(dest)
    for sid in ids or openms_ids():
        for name in OPENMS_FILES:
            target = dest / sid / name
            if not target.is_file():
                log(f"fetching  {sid}/{name}")
            fetch(f"{OPENMS_URL}/{sid}/{name}", target)
    return dest


def overlap(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Voxel-wise agreement of two binary masks.

    Dice is undefined when both masks are empty, and is given as missing then.
    """
    pred, ref = pred.astype(bool), ref.astype(bool)
    tp = int((pred & ref).sum())
    n_pred, n_ref = int(pred.sum()), int(ref.sum())
    return {
        "dice": 2 * tp / (n_pred + n_ref) if n_pred + n_ref else np.nan,
        "sensitivity": tp / n_ref if n_ref else np.nan,
        "precision": tp / n_pred if n_pred else np.nan,
    }


def lesion_detection(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Lesion-wise: a reference lesion is found if any predicted voxel touches it."""
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=bool)
    ref_lab, n_ref = ndimage.label(ref, structure=structure)
    pred_lab, n_pred = ndimage.label(pred, structure=structure)
    found = len(np.setdiff1d(np.unique(ref_lab[pred.astype(bool)]), [0]))
    true_pred = len(np.setdiff1d(np.unique(pred_lab[ref.astype(bool)]), [0]))
    return {
        "ref_lesions": int(n_ref), "pred_lesions": int(n_pred),
        "lesion_sensitivity": found / n_ref if n_ref else np.nan,
        "lesion_precision": true_pred / n_pred if n_pred else np.nan,
    }


def same_voxel_grid(a, b) -> bool:
    """Same shape, voxel size and orientation, with origins at most one voxel apart.

    The Open MS consensus masks share the T1's voxel grid, but their headers put
    the origin half a voxel away from the T1's (a centre-versus-corner difference
    in how the files were written). The dataset documents the images as
    co-registered voxel for voxel, so the masks are compared by voxel index and a
    sub-voxel origin difference in the header is accepted.
    """
    if a.shape != b.shape:
        return False
    if not np.allclose(a.affine[:3, :3], b.affine[:3, :3], atol=1e-3):
        return False
    shift_voxels = np.linalg.solve(a.affine[:3, :3], a.affine[:3, 3] - b.affine[:3, 3])
    return bool((np.abs(shift_voxels) <= 1.0).all())


def compare_subject(sid: str, folder: Path, cfg: Config, force: bool = False) -> dict:
    """Segment one Open MS patient's T1 and compare with the consensus mask."""
    import nibabel as nib

    numbers = segment.segment_subject(sid, folder / "T1W.nii.gz", cfg, force=force)
    sub = segment.subject_dir(cfg, sid)
    pred_img = nib.load(str(sub / segment.LESION_FILE))
    ref_img = nib.load(str(folder / "consensus_gt.nii.gz"))
    pred = np.asarray(pred_img.dataobj) > 0
    ref = np.asarray(ref_img.dataobj) > 0
    if not same_voxel_grid(pred_img, ref_img):
        raise PipelineError(f"{sid}: the prediction and the reference are not on the same grid")
    voxel_ml = numbers["voxel_volume_mm3"] / 1000
    row = {"subject_id": sid,
           "pred_volume_ml": float(pred.sum() * voxel_ml),
           "ref_volume_ml": float(ref.sum() * voxel_ml)}
    row.update(overlap(pred, ref))
    row.update(lesion_detection(pred, ref))
    return row


def bland_altman(pred, ref) -> dict:
    diff = np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    return {"bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def summarise(rows: pd.DataFrame) -> dict:
    """Group-level agreement from the per-patient rows."""
    from scipy import stats as sps

    pred, ref = rows["pred_volume_ml"].to_numpy(), rows["ref_volume_ml"].to_numpy()
    pearson = sps.pearsonr(pred, ref)
    spearman = sps.spearmanr(pred, ref)
    out = {"n": int(len(rows))}
    for col in ("dice", "sensitivity", "precision", "lesion_sensitivity", "lesion_precision"):
        v = rows[col].dropna()
        out[col] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "median": float(v.median()),
                    "q1": float(v.quantile(0.25)), "q3": float(v.quantile(0.75))}
    out["volume"] = {
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        "pred_mean_ml": float(pred.mean()), "ref_mean_ml": float(ref.mean()),
        "pred_median_ml": float(np.median(pred)), "ref_median_ml": float(np.median(ref)),
        "bland_altman_ml": bland_altman(pred, ref),
    }
    return out


def figures(rows: pd.DataFrame, summary: dict, folder: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    pred, ref = rows["pred_volume_ml"], rows["ref_volume_ml"]
    vol, ba = summary["volume"], summary["volume"]["bland_altman_ml"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    top = max(pred.max(), ref.max()) * 1.05
    ax.plot([0, top], [0, top], color="0.6", lw=1, ls="--")
    ax.scatter(ref, pred, s=22, color="#1f5a8a")
    ax.set(xlabel="expert consensus volume (mL)", ylabel="MindGlide T1-only volume (mL)",
           title=f"Lesion volume  r={vol['pearson_r']:.2f}, rho={vol['spearman_rho']:.2f}",
           xlim=(0, top), ylim=(0, top))

    ax = axes[1]
    ax.scatter((pred + ref) / 2, pred - ref, s=22, color="#1f5a8a")
    for y, style in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
        ax.axhline(y, color="0.4", lw=1, ls=style)
    ax.set(xlabel="mean of the two volumes (mL)", ylabel="MindGlide minus expert (mL)",
           title=f"Bland-Altman  bias {ba['bias']:+.1f} mL")

    ax = axes[2]
    ax.hist(rows["dice"].dropna(), bins=np.linspace(0, 1, 21), color="#1f5a8a")
    ax.axvline(summary["dice"]["median"], color="0.3", lw=1, ls="--")
    ax.set(xlabel="Dice", ylabel="patients", xlim=(0, 1),
           title=f"Dice  mean {summary['dice']['mean']:.2f}, median {summary['dice']['median']:.2f}")

    fig.tight_layout()
    out = folder / "openms_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out]


def report_markdown(summary: dict) -> str:
    def cell(key):
        s = summary[key]
        return f"{s['mean']:.2f} ± {s['sd']:.2f} | {s['median']:.2f} ({s['q1']:.2f} to {s['q3']:.2f})"

    vol, ba = summary["volume"], summary["volume"]["bland_altman_ml"]
    return "\n".join([
        "# MindGlide on T1 alone vs expert masks (Open MS)",
        "",
        f"{summary['n']} patients. MindGlide saw only the T1-weighted image. The reference is the",
        "multi-rater consensus mask, which was drawn with FLAIR and T2 available, so some",
        "reference lesions are faint or invisible on T1. Nothing was tuned on these data.",
        "",
        "| Measure | Mean ± SD | Median (IQR) |",
        "|---|---|---|",
        f"| Dice (voxels) | {cell('dice')} |",
        f"| Sensitivity (voxels) | {cell('sensitivity')} |",
        f"| Precision (voxels) | {cell('precision')} |",
        f"| Sensitivity (lesions) | {cell('lesion_sensitivity')} |",
        f"| Precision (lesions) | {cell('lesion_precision')} |",
        "",
        "| Lesion volume | Value |",
        "|---|---|",
        f"| Pearson r | {vol['pearson_r']:.3f} (p = {vol['pearson_p']:.2g}) |",
        f"| Spearman rho | {vol['spearman_rho']:.3f} (p = {vol['spearman_p']:.2g}) |",
        f"| Mean volume, MindGlide / expert | {vol['pred_mean_ml']:.1f} / {vol['ref_mean_ml']:.1f} mL |",
        f"| Median volume, MindGlide / expert | {vol['pred_median_ml']:.1f} / {vol['ref_median_ml']:.1f} mL |",
        f"| Bland-Altman bias (MindGlide minus expert) | {ba['bias']:+.1f} mL |",
        f"| 95% limits of agreement | {ba['loa_low']:+.1f} to {ba['loa_high']:+.1f} mL |",
        "",
        "Lesion-wise sensitivity counts a reference lesion as found when any predicted voxel",
        "touches it; lesion-wise precision is the share of predicted lesions that touch a",
        "reference lesion.",
        "",
        "Masks are compared voxel by voxel. The consensus masks' headers put the origin half a",
        "voxel away from the T1's; the dataset documents the images as co-registered on one",
        "grid, so that header difference is ignored.",
        "",
        f"Data: {OPENMS_CITATION}",
        "",
    ])


def validate_openms(cfg: Config, data_dir: Path, force: bool = False, log=print) -> dict:
    """Fetch, segment, compare, and write the report. Returns the summary."""
    fetch_openms(data_dir, log=log)
    rows = []
    for sid in openms_ids():
        row = compare_subject(sid, data_dir / sid, cfg, force=force)
        log(f"{sid}  dice {row['dice']:.3f}  volume {row['pred_volume_ml']:.1f} vs "
            f"{row['ref_volume_ml']:.1f} mL")
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = summarise(table)
    out = cfg.paths.output_dir / "validation"
    out.mkdir(parents=True, exist_ok=True)
    # per-patient rows stay with the local outputs. the summary has no patient rows
    cfg.paths.local_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.local_dir / "openms_per_patient.csv", index=False)
    (out / "openms_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "openms_report.md").write_text(report_markdown(summary), encoding="utf-8")
    figures(table, summary, out)
    log(f"wrote     {out / 'openms_report.md'}")
    return summary
