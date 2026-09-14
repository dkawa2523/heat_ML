from pathlib import Path

from external_tools.comsol_chip_cooling.comsol_runtime import provenance_path


def test_provenance_path_supports_external_data_root(tmp_path: Path) -> None:
    tool_root = tmp_path / "tool"
    inside = tool_root / "data" / "case.csv"
    outside = tmp_path / "published" / "case.csv"

    assert provenance_path(inside, tool_root) == "data/case.csv"
    assert provenance_path(outside, tool_root) == outside.resolve().as_posix()
