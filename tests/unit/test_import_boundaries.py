"""Lightweight public operations remain independent of the training runtime."""

import subprocess
import sys


def test_array_analysis_and_cae_conversion_do_not_import_torch_or_plotting() -> None:
    script = """
import sys
from celltemp.analysis import rmse
from celltemp.domain import forecast_origin_index
from celltemp.workflows.common import project_options
from external_tools.comsol_chip_cooling import dataset
assert 'torch' not in sys.modules
assert 'matplotlib' not in sys.modules
assert project_options({})['diagnostics'] is False
"""
    result = subprocess.run(  # noqa: S603 -- fixed Python script in an isolated interpreter
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_cli_help_does_not_load_numerical_or_plotting_backends() -> None:
    script = """
import sys
from celltemp.cli import main
assert 'torch' not in sys.modules
assert 'matplotlib' not in sys.modules
sys.argv = ['celltemp', '--help']
main()
"""
    result = subprocess.run(  # noqa: S603 -- fixed Python script in an isolated interpreter
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "analyze" in result.stdout
