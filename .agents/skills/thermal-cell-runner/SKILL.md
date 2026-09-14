---
name: thermal-cell-runner
description: Run and verify this repository's thermal-cell-practical setup, train, forecast, monitor, TopCell benchmark, COMSOL evaluation, and CAE report-refresh workflows. Use only inside this repository; do not use for algorithm changes or unrelated thermal projects.
---

# Thermal Cell Runner

Run the repository's existing commands and return evidence from their canonical outputs. Do not
change model code, input data, configuration, acceptance thresholds, or generated evidence unless
the user explicitly asks for that change.

## Confirm repository scope

Before running anything:

1. Resolve the repository root with `git rev-parse --show-toplevel` and work from that directory.
2. Require all of the following markers:
   - `pyproject.toml` declares the project `thermal-cell-practical`.
   - `src/celltemp/`, `quality.py`, and `README.md` exist.
   - this skill is located at `.agents/skills/thermal-cell-runner/SKILL.md` under that root.
3. Stop if the markers do not match. Never apply this runbook to another checkout by analogy.
4. Inspect `git status --short` before and after execution. Preserve existing changes; never use
   `git reset`, `git checkout --`, `git clean`, or create a commit unless separately requested.

## Choose the requested operation

Use the narrowest matching mode. If the request leaves a material choice between quickstart,
TopCell, linear COMSOL, and nonlinear/high-fidelity COMSOL, ask one concise question instead of
guessing.

- **Environment setup:** read the `README.md` execution section, then run
  `uv sync --extra dev --locked`. Do not update `uv.lock` during ordinary execution.
- **Quality gate:** run `uv run --locked python quality.py fast` for a quick check or
  `uv run --locked python quality.py pr` for the full repository gate.
- **Quickstart:** use `examples/topcell_quickstart/config.yaml`; run only the requested
  `train`, `forecast`, or `monitor` stages. Run them in that order when the user requests the
  complete quickstart.
- **TopCell benchmark:** read `benchmarks/topcell/README.md`, then run
  `uv run --locked --with-editable . python benchmarks/topcell/run.py`.
- **Linear COMSOL model evaluation:** read `external_tools/comsol_chip_cooling/README.md`, then use
  the existing dataset with the configured `train -> forecast -> monitor -> evaluate.py` flow.
- **Nonlinear or high-fidelity COMSOL evaluation:** read both
  `external_tools/comsol_chip_cooling/README.md` and
  `external_tools/comsol_chip_cooling/docs/high_fidelity_validation.md`. Use existing published
  datasets for evaluation unless the user explicitly requests new COMSOL solves.
- **CAE report refresh:** read
  `external_tools/comsol_chip_cooling/reports/cae_benchmark_report/README.md`, ensure the relevant
  benchmark outputs are current, then run its `build_report.py` command. Treat `artifact.json` and
  `source_data/` as canonical; do not recreate the ignored `report.html` unless explicitly asked
  for a portable render.

Read `docs/units_and_conventions.md` before preparing, interpreting, or changing any input,
observer setting, or physical parameter. Read only the documentation relevant to the selected
mode.

## Preflight the run

- Prefer the repository's `uv` environment and existing CLI/scripts; do not implement a wrapper
  or duplicate their logic.
- Confirm the selected config and inspect its input, artifact, output, `overwrite`, control
  convention, and device settings. Paths are relative to the config directory, not the shell CWD.
- Confirm that expected input files exist. Do not synthesize missing measurements or treat an
  empty experiment template as validation data.
- Core quickstart and TopCell work products live under ignored `work/` directories. COMSOL dataset
  generators and high-fidelity benchmark/report commands can update tracked evidence; state this
  before running them and include those changes in the final summary.
- New COMSOL solves require an explicit user request. Before a solve, inspect the selected
  script's `--help` and follow the documented license, installation, case, mesh, and `--reuse-raw`
  semantics. Never add `--reuse-raw` merely to bypass a missing installation when required raw
  tables are incomplete.
- Do not silently set `overwrite: true`, lower a benchmark criterion, shorten training, change a
  split, or select a cheaper mesh to obtain a passing result.

## Canonical core commands

Use these forms from the repository root, substituting a user-selected config when supplied:

```text
uv run --locked --with-editable . celltemp train --config examples/topcell_quickstart/config.yaml
uv run --locked --with-editable . celltemp forecast --config examples/topcell_quickstart/config.yaml
uv run --locked --with-editable . celltemp monitor --config examples/topcell_quickstart/config.yaml
```

For the existing linear COMSOL dataset:

```text
uv run --locked --with-editable . celltemp train --config external_tools/comsol_chip_cooling/config.yaml
uv run --locked --with-editable . celltemp forecast --config external_tools/comsol_chip_cooling/config.yaml
uv run --locked --with-editable . celltemp monitor --config external_tools/comsol_chip_cooling/config.yaml
uv run --locked --with-editable . python external_tools/comsol_chip_cooling/evaluate.py
```

For existing high-fidelity data and the canonical evidence artifact:

```text
uv run --locked --with-editable . python external_tools/comsol_chip_cooling/benchmark_high_fidelity.py
uv run --locked --with-editable . python external_tools/comsol_chip_cooling/reports/cae_benchmark_report/build_report.py
```

Use documented dot-key CLI overrides only when the user explicitly requests them. Do not persist
temporary overrides back into YAML unless asked.

## Execute and verify

Run dependent stages sequentially and stop at the first nonzero exit code. Do not claim success
from file existence alone.

- **Train:** inspect `metrics_summary.json`, `metrics_by_case.csv`, `split.csv`, and artifact
  `metadata.json`. Report causal and conditional metrics separately; model selection uses causal
  validation RMSE.
- **Forecast:** inspect `forecast_summary.csv`, `forecast_coverage.csv`, and `run_manifest.json`.
  Surface control, predicted-temperature, timestep, horizon, and slew OOD warnings. An OOD warning
  is not automatically a command failure, but it limits interpretation.
- **Monitor:** inspect `monitor_summary.csv` and `run_manifest.json`. Report the bias gauge,
  innovation/NIS evidence, disturbance basis, and missing-data behavior relevant to the request.
- **Benchmark/evaluation:** parse the published summary JSON and require every declared check or
  workflow status to pass. Preserve a failing exit code and explain the failed checks without
  modifying thresholds or inputs.
- **Report:** verify the rebuilt artifact timestamp, case inventory, datasets, and input hashes.

Run `quality.py` after execution only when the user requested code quality validation or code was
changed. Do not add redundant tests or diagnostics for an ordinary model run.

## Interpret results conservatively

- Forecast intervals cover latent-state and configured process uncertainty only. They exclude
  parameter, future-input, measurement, and model-form uncertainty and are not empirical coverage
  guarantees.
- A high-fidelity workflow pass is not strict physical validation. Report
  `data_quality.strict_validation_status`, mesh and temporal qualification, experiment status, and
  permitted use exactly as recorded.
- Do not use volume-average benchmark error as a hotspot safety limit or product acceptance value.
- Keep benchmark/model-gap cases visible; do not average them into qualified forecast performance.

## Return the run record

Summarize the exact command and config, exit status, output directory, principal causal metrics,
coverage/OOD warnings, validation qualification, and files changed by the run. Link the canonical
summary, manifest, or report artifact. If blocked, identify the missing input, environment, license,
or explicit mode choice and leave existing outputs untouched.
