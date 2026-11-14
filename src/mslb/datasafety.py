"""Keep patient data out of git.

Checks file paths (and sizes) against the rules below and reports every one that
breaks them. Runs three ways: the pre-commit hook calls this file directly with
--staged, `mslb check-data` runs it over everything tracked, and the tests call
check_paths. Standard library only, so the hook works before anything is installed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# imaging, arrays, model weights, spreadsheets, stats-package files
BLOCKED_SUFFIXES = (
    ".nii", ".nii.gz", ".mgz", ".mgh", ".dcm", ".ima", ".nrrd", ".mha", ".mhd",
    ".img", ".hdr", ".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5", ".mat",
    ".xlsx", ".xls", ".sav", ".rds", ".rdata", ".parquet", ".feather", ".pkl",
)

# delimited tables are only allowed where the empty schema templates live
TABLE_SUFFIXES = (".csv", ".tsv")
TABLE_DIRS = ("templates",)

# pipeline inputs and outputs, and local tool state
BLOCKED_DIRS = ("data", "outputs", "cache", "viewer/public/data", ".claude")

# the local config carries local paths. config.example.yaml is the committed one
BLOCKED_FILES = ("config.yaml",)

MAX_BYTES = 1_000_000
# generated lockfiles can be large and carry no data
SIZE_EXEMPT = ("viewer/package-lock.json",)


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _under(path: PurePosixPath, folder: str) -> bool:
    parts, want = path.parts, PurePosixPath(folder).parts
    return parts[: len(want)] == want


def check_path(path: str, size: int | None = None) -> Violation | None:
    """One repo-relative path, forward or back slashes. None when it may be committed."""
    posix = PurePosixPath(path.replace("\\", "/"))
    text = posix.as_posix()
    lower = text.lower()

    for folder in BLOCKED_DIRS:
        if _under(posix, folder):
            return Violation(text, f"is under {folder}/, which holds local data")
    if text in BLOCKED_FILES:
        return Violation(text, "is the local config, commit config.example.yaml instead")
    for suffix in BLOCKED_SUFFIXES:
        if lower.endswith(suffix):
            return Violation(text, f"{suffix} files are never committed")
    for suffix in TABLE_SUFFIXES:
        if lower.endswith(suffix) and not any(_under(posix, d) for d in TABLE_DIRS):
            return Violation(text, f"{suffix} files are only allowed in {', '.join(TABLE_DIRS)}/")
    if size is not None and size > MAX_BYTES and text not in SIZE_EXEMPT:
        return Violation(text, f"is {size / 1e6:.1f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit")
    return None


def check_paths(items) -> list[Violation]:
    """items: (path, size-or-None) pairs. Every violation, in the order given."""
    found = []
    for path, size in items:
        v = check_path(path, size)
        if v is not None:
            found.append(v)
    return found


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8"
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or f"git {' '.join(args)} failed")
    return done.stdout


def repo_root(start: Path | None = None) -> Path:
    return Path(_git(start or Path.cwd(), "rev-parse", "--show-toplevel").strip())


def staged_items(root: Path) -> list[tuple[str, int | None]]:
    """Files added, copied, modified or renamed in the index, with their staged size."""
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    items = []
    for name in filter(None, names.split("\0")):
        try:
            size = int(_git(root, "cat-file", "-s", f":{name}").strip())
        except (RuntimeError, ValueError):
            size = None
        items.append((name, size))
    return items


def tracked_items(root: Path) -> list[tuple[str, int | None]]:
    """Every tracked file, with its size in the working tree."""
    names = _git(root, "ls-files", "-z")
    items = []
    for name in filter(None, names.split("\0")):
        p = root / name
        items.append((name, p.stat().st_size if p.is_file() else None))
    return items


def run(staged: bool = False, root: Path | None = None) -> int:
    """Print the violations and return an exit code: 0 clean, 1 violations, 2 no git."""
    try:
        root = root or repo_root()
        items = staged_items(root) if staged else tracked_items(root)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"data check could not run: {exc}", file=sys.stderr)
        return 2
    what = "staged" if staged else "tracked"
    violations = check_paths(items)
    if violations:
        print(f"data check failed, {len(violations)} {what} file(s) may not be committed:",
              file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print(f"data check ok: {len(items)} {what} file(s)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check that no data files are in git.")
    parser.add_argument("--staged", action="store_true",
                        help="check what is staged for commit instead of everything tracked")
    args = parser.parse_args(argv)
    return run(staged=args.staged)


if __name__ == "__main__":
    sys.exit(main())
