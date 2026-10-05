# TopCell quickstart

このディレクトリだけで、熱時系列解析と、同じ熱RC artifactを使う学習・open-loop予測を
試せます。実測の逐次状態推定が必要な場合だけmonitorを追加します。
CSVはすべて自己完結形式です。
自分のdataへ適用するときは[config.minimal.yaml](config.minimal.yaml)の少ない設定から始め、
学習・observerの詳細は[config.yaml](config.yaml)を参照してください。最小例の未指定学習は既定80 epochsです。
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
| train | `work/outputs/runs/thermal_network_demo/metrics_summary.json`、`diagnostics/model_comparison.csv`、`diagnostics/figures/test_prediction_timeseries.png`、`diagnostics/figures/test_prediction_parity.png`、`diagnostics/thermal_paths.csv` |
| forecast | `work/outputs/forecast/forecast_summary.csv`、`cases/<case_id>.csv`、`diagnostics/figures/forecast_<case_id>.png`、`diagnostics/energy_balance.csv` |

forecastの主要図は予測温度・95%状態区間・時間変化するcommand・sensor spanを示します。熱収支図は
`diagnostics/figures/energy_balance_<case_id>.png`です。`recipe_case`は時間変化する入熱・冷却の操作例であり、
特定solverや実機の妥当化結果ではありません。

`config.yaml`からの相対パスで`system.yaml`と`data/`を参照します。その他の生成物も`work/`だけへ保存され、
再実行時は処理が完了した結果だけが置換されます。

`config.yaml`は任意診断を有効にしています。`config.minimal.yaml`は基本CSVとartifactだけを保存し、
予測図・熱収支を必要とするときだけ`project.diagnostics: true`を追加します。最小例の成果物は
`work/outputs/runs/thermal_network_minimal/`と`work/outputs/forecast_minimal/`へ保存されます。
forecast/monitorの列は`sensor.<名前>.<量>`、`node.<名前>.<量>`、`control.<名前>.<量>`です。
温度は既定degC、時間はs、温度差はKです。
