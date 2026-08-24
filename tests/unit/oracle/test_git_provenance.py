"""The manifest's dirty flag must mean "the code was uncommitted" (ADR 0019).

Both the ADR 0017 and ADR 0018 runs shipped `git_dirty: true`, so the reports
name a commit that never produced those numbers. The flag is the guard against
a fourth occurrence, and it only works if it does not cry wolf: a run writes
its own output under `results/`, and that output cannot change what the code
did.
"""

from __future__ import annotations

import subprocess

import pytest

from pos.oracle.run_helpers import dirty_entries, git_dirty


@pytest.fixture()
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


class TestGitDirty:
    def test_clean_tree_is_clean(self, repo):
        assert git_dirty(repo) is False

    def test_untracked_run_output_does_not_count(self, repo):
        out = repo / "results" / "experiments" / "run"
        out.mkdir(parents=True)
        (out / "summary.csv").write_text("a,b\n1,2\n")
        assert dirty_entries(repo) == []
        assert git_dirty(repo) is False

    def test_untracked_source_still_counts(self, repo):
        (repo / "new_module.py").write_text("y = 2\n")
        assert git_dirty(repo) is True

    def test_modified_tracked_file_counts(self, repo):
        (repo / "code.py").write_text("x = 2\n")
        assert git_dirty(repo) is True

    def test_modified_file_under_results_still_counts(self, repo):
        """Only *untracked* output is exempt; editing a committed run is not."""
        out = repo / "results"
        out.mkdir()
        (out / "kept.csv").write_text("a\n1\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "keep"], cwd=repo, check=True)
        (out / "kept.csv").write_text("a\n2\n")
        assert git_dirty(repo) is True
