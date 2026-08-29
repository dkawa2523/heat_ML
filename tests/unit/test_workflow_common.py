"""Output boundaries shared by the three filesystem workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from celltemp.workflows.common import output_target, staged_output_directory


def test_output_target_requires_an_explicit_absolute_path_outside_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external-results"
    target, _ = output_target({"output_dir": str(external)}, project)
    assert target == external.resolve()

    with pytest.raises(ValueError, match="relative output_dir"):
        output_target({"output_dir": "../external-results"}, project)
    with pytest.raises(ValueError, match="root or an ancestor"):
        output_target({"output_dir": str(tmp_path)}, project)


def test_staged_output_replaces_only_after_success(tmp_path: Path) -> None:
    target = tmp_path / "results"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    def fail_during_staging() -> None:
        with staged_output_directory(target, overwrite=True) as staging:
            (staging / "new.txt").write_text("incomplete", encoding="utf-8")
            raise RuntimeError("processing failed")

    with pytest.raises(RuntimeError, match="processing failed"):
        fail_during_staging()
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"

    with staged_output_directory(target, overwrite=True) as staging:
        (staging / "new.txt").write_text("complete", encoding="utf-8")
    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text(encoding="utf-8") == "complete"
    assert not list(tmp_path.glob(".results-backup-*"))
