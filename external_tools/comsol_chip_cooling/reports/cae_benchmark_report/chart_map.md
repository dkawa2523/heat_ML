# Chart map and report QA notes

Generated: 2026-08-31T01:13:27+00:00

Audience: technical. Delivery mode: portable HTML from canonical artifact.json.

## Required-structure mapping

1. Title → `title`
2. Technical summary → `technical_summary` + headline metric strip
3. Key findings with visual evidence → linear, nonlinear, mesh, and high-fidelity result sections
4. Scope/data/definitions → `problem_boundary`, case catalogs, metric definition table
5. Methodology → `metrics_method`
6. Limitations/uncertainty/robustness → `limitations`
7. Recommended next steps → `next_steps`
8. Further questions → `further_questions`

## Visual contracts

| Visual | Question | Family/type | Fields | Data sufficiency | Palette policy | Supported claim |
|---|---|---|---|---|---|---|
| coverage_chart | How are case IDs distributed by layer and role? | composition / stacked bar | layer, role, cases | 9 nonzero groups | relaxed categorical | The suite is layered rather than one undifferentiated benchmark. |
| linear_forecast_chart | Which linear forecast is hardest? | comparison / horizontal bar | case_id, rmse_k | 8 cases | single-root | Short pulse is worst but all cases are far below the criterion. |
| linear_monitor_chart | Which monitoring cases trigger alerts? | comparison / horizontal bar | case_id, alert_rate_percent | 5 cases | single-root | Alerts are confined to injected abrupt offset/heat cases. |
| nonlinear_envelope_chart | How far does each nonlinear role extend thermally? | comparison / horizontal bar | role, max_chip_c | 4 roles | single-root | Forecast cases intentionally extend beyond the training thermal envelope. |
| radiation_pairs_chart | How does the medium-fidelity radiation gap vary by condition? | comparison / horizontal bar | pair, abs_terminal_chip_delta_k | 5 matched pairs | single-root | Hot/low-flow and hot-start show the largest screening gaps. |
| mesh_convergence_chart | How does chip temperature move with mesh profile? | comparison / grouped bar | mesh_label, case_label, chip_average_c | 4 profiles × 2 cases | hard two-root | Global mesh is inadequate and convergence differs by operating point. |
| hf_error_chart | Are high-fidelity model errors smaller than the mesh proxy? | benchmark / grouped bar | case_label, metric, value_k | 2 cases × 3 metrics | categorical (three semantic metrics) | All model errors are below the conservative adjacent-mesh difference. |
| hv01_trace_chart | Does the model track the non-radiating composite transient? | trend / line | time, truth_chip, predicted_chip | 12 ordered points | hard two-root | HV01 temperature trajectory is closely tracked. |
| hv02_trace_chart | Does error grow when radiation is present? | trend / line | time, truth_chip, predicted_chip | 12 ordered points | hard two-root | Error grows in the coupled hot/low-flow phase. |
| hf_radiation_trace_chart | Does the model reproduce the paired radiation-only delta? | trend / line | time, truth_delta_chip_k, predicted_delta_chip_k | 12 paired points | hard two-root | Direction matches but amplitude is almost absent in the model. |

The three line charts are retained because they answer distinct temporal questions on the same 12-point experimental design. Bar-family repetition is intentional: the other questions are categorical comparisons, not continuous trends.

## Evidence gaps kept visible

- NF01–NF09 open-loop model metrics are not present in the current benchmark outputs.
- NM01–NM03 observer metrics are not present in the current benchmark outputs.
- No measured experiment values have been supplied; templates are not counted as validation.
- High-fidelity strict mesh convergence and transient time-step convergence are not complete.
