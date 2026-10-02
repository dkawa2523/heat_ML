"""One published directory for benchmark validation figures."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_FIGURE_DIRECTORY = REPOSITORY_ROOT / "docs" / "validation_figures"


def validation_figure_reference(filename: str) -> str:
    """Return the repository-relative path stored in benchmark summaries."""
    return (Path("docs") / "validation_figures" / filename).as_posix()
