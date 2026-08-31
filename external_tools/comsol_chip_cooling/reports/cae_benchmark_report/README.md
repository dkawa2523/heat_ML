# CAE benchmark report artifact

Primary deliverable: `report.html`.

- `artifact.json`: canonical bounded report input.
- `source_data/`: normalized snapshot tables used by the report.
- `chart_map.md`: chart contracts, structure mapping, and explicit evidence gaps.
- `build_report.py`: deterministic normalization and artifact authoring.

Rebuild the artifact from repository root:

```powershell
.venv\Scripts\python.exe external_tools/comsol_chip_cooling/reports/cae_benchmark_report/build_report.py
```

Then package and verify `artifact.json` with the Data Analytics portable report builder. The script consumes the published CAE datasets plus the latest ignored `work/evaluation` and model-run outputs; rerun the corresponding benchmark workflows first if those work products are absent.
