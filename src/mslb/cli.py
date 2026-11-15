"""Command line: `mslb run -c config.yaml` is the whole pipeline."""
from __future__ import annotations

import argparse
import sys

from mslb import __version__, datasafety
from mslb.config import check_inputs, load
from mslb.errors import PipelineError

# what a full run does, in order. each entry is (stage, what it produces)
STAGES = [
    ("inputs", "T1 scans matched to the clinical and FC tables"),
    ("segment", "lesion mask, lesion volume and count"),
    ("register", "T1 to template registration"),
    ("icv", "head-size scaling factor"),
    ("tracts", "lesion volume inside each tract"),
    ("tables", "per-subject table (local only)"),
    ("stats", "shareable group summary"),
]


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="path to the config file (default: config.yaml)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mslb",
        description="MS lesion burden from T1-weighted MRI and its associations with "
                    "functional connectivity, urodynamic and symptom data.",
    )
    parser.add_argument("--version", action="version", version=f"mslb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    run = sub.add_parser("run", help="run the whole pipeline")
    _add_config(run)
    run.add_argument("--force", action="store_true", help="redo stages that already have output")
    run.add_argument("--dry-run", action="store_true",
                     help="check the config and inputs, list the stages, and stop")

    show = sub.add_parser("show-config", help="print the config with defaults filled in")
    _add_config(show)

    fake = sub.add_parser("make-fake-data", help="write fake clinical and FC tables for testing")
    _add_config(fake)
    fake.add_argument("--n", type=int, default=24,
                      help="number of subjects when the scans folder is empty (default: 24)")
    fake.add_argument("--seed", type=int, default=0)
    fake.add_argument("--force", action="store_true",
                      help="overwrite the clinical and FC files named in the config")

    check = sub.add_parser("check-data", help="check that no data files are tracked by git")
    check.add_argument("--staged", action="store_true", help="check staged files only")
    return parser


def _run(args) -> int:
    cfg = load(args.config)
    problems = check_inputs(cfg)
    print(f"config    {cfg.source}  ({cfg.digest()})")
    print(f"analyses  {', '.join(cfg.enabled_analyses()) or 'none'}")
    for name, what in STAGES:
        print(f"  {name:9s} {what}")
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("dry run: config and inputs are fine")
        return 0
    print("error: the stages are not wired up yet, only --dry-run works", file=sys.stderr)
    return 3


def _show_config(args) -> int:
    import yaml

    cfg = load(args.config)
    print(yaml.safe_dump(cfg.as_dict(), sort_keys=False), end="")
    return 0


def _make_fake_data(args) -> int:
    from mslb import fakedata, inputs

    cfg = load(args.config)
    targets = [cfg.paths.clinical_csv, cfg.paths.fc_csv]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.force:
        # the same paths hold the real tables on a real run, so never replace them quietly
        print(f"error: not overwriting {', '.join(existing)}. pass --force if these are fake.",
              file=sys.stderr)
        return 2
    ids = list(inputs.discover_t1(cfg.paths.t1_dir)) if cfg.paths.t1_dir.is_dir() else []
    if ids:
        print(f"subjects  {len(ids)} scans found in {cfg.paths.t1_dir}")
    else:
        ids = fakedata.default_ids(args.n)
        print(f"subjects  {len(ids)} made-up IDs (no scans in {cfg.paths.t1_dir})")
    burden = None
    imaging = cfg.paths.local_dir / "imaging.csv"
    if imaging.is_file():
        import pandas as pd

        table = pd.read_csv(imaging, dtype={"subject_id": str}).set_index("subject_id")
        if set(ids) <= set(table.index) and "lesion_volume_ml" in table:
            burden = table["lesion_volume_ml"].to_dict()
            print(f"burden    planted effects follow lesion volume in {imaging}")
    if burden is None:
        print("burden    no imaging table yet, planted effects follow a random stand-in")
    clinical, fc = fakedata.make_fake(ids, seed=args.seed, burden=burden)
    for path, frame in zip(targets, (clinical, fc)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        print(f"wrote     {path}  ({len(frame)} rows, FAKE)")
    return 0


def _check_data(args) -> int:
    return datasafety.run(staged=args.staged)


COMMANDS = {
    "run": _run,
    "show-config": _show_config,
    "make-fake-data": _make_fake_data,
    "check-data": _check_data,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
