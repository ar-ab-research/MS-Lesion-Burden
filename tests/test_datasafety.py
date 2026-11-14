import shutil
import subprocess
from pathlib import Path

import pytest

from mslb import datasafety
from mslb.datasafety import MAX_BYTES, check_path, check_paths

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [
    "sub-01_T1w.nii.gz",
    "anywhere/deep/scan.nii",
    "src/mslb/weights.pt",
    "docs/figure_data.npy",
    "notes/cohort.xlsx",
    "SCAN.NII.GZ",                      # case does not get around it
    "tests\\fixtures\\t1.nii.gz",       # nor do windows separators
    "data/t1/readme.txt",               # anything under data/, whatever its type
    "outputs/shareable/group_summary.md",
    "cache/atlas/JHU-tracts.xml",
    "viewer/public/data/sub-01/meta.json",
    ".claude/launch.json",
    "config.yaml",
    "clinical.csv",
    "tests/fixtures/fc.csv",
    "docs/table.tsv",
])
def test_blocked(path):
    assert check_path(path) is not None


@pytest.mark.parametrize("path", [
    "templates/clinical_template.csv",
    "templates/fc_template.csv",
    "config.example.yaml",
    "src/mslb/config.py",
    "src/mslb/datasafety.py",
    "docs/config.yaml",                 # only the root config.yaml is the local one
    "viewer/src/main.js",
    "viewer/public/favicon.svg",
    "database/notes.md",                # data/ means the folder, not the prefix
    "PLAN.md",
])
def test_allowed(path):
    assert check_path(path) is None


def test_size_limit():
    assert check_path("docs/figure.png", MAX_BYTES) is None
    assert check_path("docs/figure.png", MAX_BYTES + 1) is not None
    assert check_path("viewer/package-lock.json", MAX_BYTES * 3) is None


def test_check_paths_reports_every_violation_with_a_reason():
    found = check_paths([("a.nii.gz", 10), ("src/ok.py", 10), ("data/x.txt", None)])
    assert [v.path for v in found] == ["a.nii.gz", "data/x.txt"]
    assert all(v.reason for v in found)


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "work")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    return tmp_path


@needs_git
def test_staged_scan_catches_a_force_added_scan(repo, capsys):
    (repo / "ok.py").write_text("x = 1\n")
    (repo / "scan.nii.gz").write_bytes(b"\x1f\x8b fake")
    _git(repo, "add", "-f", "ok.py", "scan.nii.gz")
    assert datasafety.run(staged=True, root=repo) == 1
    assert "scan.nii.gz" in capsys.readouterr().err


@needs_git
def test_staged_scan_passes_clean_files_and_ignores_deletions(repo):
    (repo / "ok.py").write_text("x = 1\n")
    (repo / "old.csv").write_text("a,b\n")
    _git(repo, "add", "-f", "ok.py", "old.csv")
    _git(repo, "commit", "-q", "-m", "seed")
    # taking a stray table back out must not be blocked by the table rule
    _git(repo, "rm", "-q", "old.csv")
    assert datasafety.run(staged=True, root=repo) == 0


@needs_git
def test_tracked_scan_catches_a_committed_table_and_a_big_file(repo, capsys):
    (repo / "clinical.csv").write_text("subject_id\n")
    (repo / "big.bin").write_bytes(b"0" * (MAX_BYTES + 1))
    _git(repo, "add", "-f", "clinical.csv", "big.bin")
    _git(repo, "commit", "-q", "-m", "seed")
    assert datasafety.run(staged=False, root=repo) == 1
    err = capsys.readouterr().err
    assert "clinical.csv" in err and "big.bin" in err


def test_when_git_cannot_answer_it_is_an_error_not_a_pass(monkeypatch, capsys):
    # a guard that cannot look must not report clean
    def no_repo():
        raise RuntimeError("not a git repository")

    monkeypatch.setattr(datasafety, "repo_root", no_repo)
    assert datasafety.run() == 2
    assert "could not run" in capsys.readouterr().err


@needs_git
def test_this_repo_tracks_no_data():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert check_paths(datasafety.tracked_items(ROOT)) == []
