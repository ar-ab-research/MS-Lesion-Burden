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


def _check_data(args) -> int:
    return datasafety.run(staged=args.staged)


COMMANDS = {
    "run": _run,
    "show-config": _show_config,
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
