# TopCell quickstart

このディレクトリだけで、熱時系列解析と、同じ熱RC artifactを使う学習・open-loop予測を
試せます。実測の逐次状態推定が必要な場合だけmonitorを追加します。
CSVはすべて自己完結形式です。
以下はPowerShell、macOS、Linuxで共通の`uv`コマンドです。

```powershell
uv run --locked --with-editable . celltemp analyze --config examples/topcell_quickstart/config.yaml
uv run --locked --with-editable . celltemp train --config examples/topcell_quickstart/config.yaml
uv run --locked --with-editable . celltemp forecast --config examples/topcell_quickstart/config.yaml
```

任意のmonitor:

```powershell
uv run --locked --with-editable . celltemp monitor --config examples/topcell_quickstart/config.yaml
```

最初に読む成果物は次のとおりです。

| 処理 | 成果物 |
|---|---|
| analyze | `work/outputs/analysis/summary.json`、`case_metrics.csv`、`figures/<case_id>.png` |
| train | `work/outputs/runs/thermal_network_demo/metrics_summary.json`、`model_comparison.csv`、`figures/test_prediction_timeseries.png`、`figures/test_prediction_parity.png`、`thermal_paths.csv` |
| forecast | `work/outputs/forecast/forecast_summary.csv`、`figures/forecast_<case_id>.png`、`energy_balance.csv` |

forecastの主要図は予測温度・95%状態区間・時間変化するcommand・sensor spanを示します。熱収支図は
`figures/energy_balance_<case_id>.png`です。`recipe_case`は時間変化する入熱・冷却の操作例であり、
特定solverや実機の妥当化結果ではありません。

`config.yaml`からの相対パスで`system.yaml`と`data/`を参照します。その他の生成物も`work/`だけへ保存され、
再実行時は処理が完了した結果だけが置換されます。
