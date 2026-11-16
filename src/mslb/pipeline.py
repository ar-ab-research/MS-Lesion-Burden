"""The whole run, start to finish: scans and tables in, group summary out."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from mslb import __version__, atlas, icv, inputs, register, report, schemas, segment, stats, tracts
from mslb.config import Config
from mslb.errors import PipelineError

IMAGING_CSV = "imaging.csv"
SUBJECTS_CSV = "subjects.csv"


def tract_names(cfg: Config) -> list[str]:
    return list(cfg.tracts.primary) + list(cfg.tracts.secondary)


def add_normalized(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add head-size-adjusted volumes. Rows that failed registration QC get none.

    A registration that failed gives a wrong head-size factor and misplaced tracts,
    so for those rows the factor, the adjusted volumes and the tract columns are
    blanked rather than carried into the statistics.
    """
    out = table.copy()
    bad = ~out["reg_ok"].astype(bool)
    tract_cols = [c for c in out.columns if c.startswith("tract_")]
    out.loc[bad, ["head_scale", "etiv_ml", *tract_cols]] = np.nan
    out["lesion_volume_norm_ml"] = out["lesion_volume_ml"] * out["head_scale"]
    for name in tract_names(cfg):
        if f"tract_{name}_ml" in out:
            out[f"tract_{name}_norm_ml"] = out[f"tract_{name}_ml"] * out["head_scale"]
    return out


def subject_table(imaging: pd.DataFrame, clinical: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """One row per scanned subject, with whatever the two tables hold for them."""
    table = imaging.merge(clinical, on="subject_id", how="left")
    return table.merge(schemas.fc_wide(fc), on="subject_id", how="left")


def process_subject(sid: str, t1: Path, cfg: Config, force: bool, template_ml: float) -> dict:
    """Segment, register and measure one subject. Returns one row of the imaging table."""
    inputs.check_t1(t1)
    row = {"subject_id": sid}
    seg = segment.segment_subject(sid, t1, cfg, force=force)
    row.update({k: seg[k] for k in ("lesion_volume_ml", "lesion_count",
                                    "lesion_volume_unfiltered_ml", "labelled_volume_ml")})
    reg = register.register_subject(sid, t1, cfg, force=force)
    row.update({k: reg[k] for k in ("head_scale", "reg_similarity", "reg_ok")})
    row["etiv_ml"] = template_ml / reg["head_scale"]
    row["icv_method"] = "affine_scaling"
    loads = tracts.tract_subject(sid, t1, cfg)
    row.update({k: v for k, v in loads.items() if k != "subject_id"})
    return row


def versions() -> dict:
    out = {"mslb": __version__, "python": platform.python_version()}
    for name in ("mindglide", "antspyx", "torch", "numpy", "scipy", "pandas", "statsmodels",
                 "nibabel", "templateflow"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run(cfg: Config, force: bool = False, log=print) -> dict:
    """Run every stage. Returns the manifest that is also written to the output folder."""
    started = time.time()
    timings = {}

    def stage(name: str) -> None:
        timings[name] = round(time.time() - stage.t0, 1)
        stage.t0 = time.time()

    stage.t0 = started

    # read and check everything that can be wrong before any slow step
    scans = inputs.discover_t1(cfg.paths.t1_dir)
    if not scans:
        raise PipelineError(f"no T1 scans found in {cfg.paths.t1_dir}")
    clinical = schemas.load_clinical(cfg.paths.clinical_csv, cfg)
    fc = schemas.load_fc(cfg.paths.fc_csv)
    matched = inputs.match_ids(set(scans), set(clinical["subject_id"]), set(fc["subject_id"]))
    log(f"inputs    {len(scans)} scans, {len(clinical)} clinical rows, "
        f"{fc['subject_id'].nunique()} subjects with FC, {len(matched['complete'])} with all three")
    for key in ("scan_without_clinical", "scan_without_fc", "clinical_without_scan", "fc_without_scan"):
        if matched[key]:
            log(f"  note: {key.replace('_', ' ')}: {', '.join(matched[key])}")
    template_ml = icv.mask_volume_ml(atlas.template_paths(cfg)["mask"])
    atlas.jhu_paths(cfg, log=log)
    stage("inputs")

    rows, failed = [], {}
    for i, (sid, t1) in enumerate(scans.items(), 1):
        t0 = time.time()
        try:
            rows.append(process_subject(sid, t1, cfg, force, template_ml))
            log(f"subject   {i}/{len(scans)} {sid}  {time.time() - t0:.0f} s")
        except PipelineError as exc:
            # one unusable scan should not cost the rest of the cohort
            failed[sid] = str(exc)
            log(f"subject   {i}/{len(scans)} {sid}  FAILED: {exc}")
    if not rows:
        raise PipelineError("every subject failed, see the messages above")
    stage("subjects")

    imaging = pd.DataFrame(rows)
    if cfg.icv.method == "external_csv":
        ext = icv.external_scales(cfg, list(imaging["subject_id"])).set_index("subject_id")
        for col in ("etiv_ml", "head_scale", "icv_method"):
            imaging[col] = ext[col].reindex(imaging["subject_id"]).to_numpy()
    imaging = add_normalized(imaging, cfg)
    qc_failed = sorted(imaging.loc[~imaging["reg_ok"].astype(bool), "subject_id"])
    if qc_failed:
        log(f"  note: registration QC failed, no head-size or tract values: {', '.join(qc_failed)}")

    local = cfg.paths.local_dir
    local.mkdir(parents=True, exist_ok=True)
    imaging.to_csv(local / IMAGING_CSV, index=False)
    table = subject_table(imaging, clinical, fc)
    table.to_csv(local / SUBJECTS_CSV, index=False)
    log(f"wrote     {local / SUBJECTS_CSV}  (per subject, stays on this machine)")
    stage("tables")

    tests = stats.run_analyses(table, cfg)
    counts = {
        "scans": len(scans), "processed": len(rows), "failed": len(failed),
        "registration_qc_failed": len(qc_failed),
        "with_clinical": int(table["subject_id"].isin(clinical["subject_id"]).sum()),
        "with_fc": int(table["subject_id"].isin(fc["subject_id"]).sum()),
    }
    manifest = {
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_digest": cfg.digest(),
        "settings": {k: v for k, v in cfg.as_dict().items() if k != "paths"},
        "versions": versions(),
        "counts": counts,
        "seconds": timings,
    }
    written = report.write(table, tests, cfg, manifest)
    stage("stats")
    manifest["seconds"] = timings
    manifest["seconds"]["total"] = round(time.time() - started, 1)
    # the manifest with subject ids of failures stays local. the shareable copy has counts only
    (local / "run_manifest.json").write_text(
        json.dumps({**manifest, "failed_subjects": failed, "registration_qc_failed": qc_failed,
                    "id_matching": matched}, indent=2), encoding="utf-8")
    for path in written:
        log(f"wrote     {path}")
    return manifest
