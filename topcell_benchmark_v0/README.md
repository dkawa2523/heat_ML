# TopCell-ICP-Thermal Benchmark v0 + thermal-cell v6

半導体製造装置のICP/RIE系プラズマチャンバー上部Cellを想定した、温度時系列予測コード v6 の実行評価用パッケージです。

このZIPには以下が含まれます。

```text
1. v6コード本体
2. TopCell-ICP-Thermal Benchmark v0 の合成CAEデータ
3. Graph-RC用 cell_nodes.csv / cell_edges.csv
4. forecast用 schedule / conditions
5. monitor用 疑似実測ログ
6. 3点センサー用config
7. 高plasma外挿評価用config
8. データ再生成スクリプト
```

## できること

- 定数条件CAEからの温度応答学習
- 時間変化する `brine / heater / plasma` 入力の学習・推論
- `linear_rc / mlp / cnn1d / gru / lstm / tcn / thermal_state_space / graph_rc` の比較
- Graph-RCで既知のCell位置関係・熱抵抗表を利用
- 3点 / 4点など少数計測点の学習
- forecast: 条件CSVとschedule_csvからopen-loop時系列予測
- monitor: 実測ログを使ったone-step残差監視
- residuals: monitor出力から残差long table、summary、offset補正候補を生成

Neural ODE / PINN / DeepONet / FNO は、この問題では対象外です。

## インストール

```bash
pip install -e .
```

インストールせずに使う場合は、各コマンドの前に `PYTHONPATH=src` を付けてください。

## まず動かす

### 1. Graph-RC学習

```bash
celltemp train --config configs/config_train.yaml
```

短時間確認だけなら:

```bash
celltemp train --config configs/config_train.yaml train.epochs=3 project.run_name=smoke_graph_rc
```

### 2. forecast推論

```bash
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc_topcell_v0/model_package \
  prediction.mode=forecast
```

### 3. monitor推論

```bash
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc_topcell_v0/model_package \
  prediction.mode=monitor \
  prediction.input_table=data/pred/monitor_cases.csv \
  prediction.output_dir=outputs/monitor_graph_rc
```

### 4. 残差解析

```bash
celltemp residuals --config configs/config_pred.yml \
  residuals.input_dir=outputs/monitor_graph_rc \
  residuals.output_dir=outputs/residual_analysis
```

### 5. モデル比較

```bash
celltemp compare --config configs/config_train.yaml
```

## モデル比較例

```bash
celltemp train --config configs/config_train.yaml model.name=thermal_state_space project.run_name=tss_topcell_v0
celltemp train --config configs/config_train.yaml model.name=graph_rc project.run_name=graph_rc_topcell_v0
celltemp train --config configs/config_train.yaml model.name=tcn project.run_name=tcn_topcell_v0
celltemp train --config configs/config_train.yaml model.name=gru project.run_name=gru_topcell_v0
celltemp compare --config configs/config_train.yaml
```

## 3点センサー評価

```bash
celltemp train --config configs/config_train_3pt.yaml
```

このconfigでは `sensor_cols: [Center, middle, edge]` を使い、Graph-RCは3点サブグラフで動きます。

## 高plasma外挿評価

```bash
celltemp train --config configs/config_train_holdout_plasma.yaml
```

`split.method=holdout_max` により、最大plasma条件がtest側になります。

## データ再生成

```bash
python scripts/generate_topcell_benchmark.py
```

## データ内容

詳細は以下を参照してください。

```text
docs/topcell_benchmark_problem.md
```

## 主な成果物

```text
outputs/runs/<run>/train_history.csv
outputs/runs/<run>/metrics/rollout_summary.csv
outputs/runs/<run>/plots/*.png
outputs/runs/<run>/graph/learned_conductance.csv
outputs/runs/<run>/graph/graph_diagnostics.csv
outputs/predictions_graph_rc/*.csv
outputs/monitor_graph_rc/*_monitor.csv
outputs/residual_analysis/residual_long.csv
outputs/residual_analysis/residual_summary.csv
outputs/residual_analysis/offset_correction.json
outputs/model_leaderboard.csv
```

## 注意

このデータは実装・評価用の合成ベンチマークです。半導体プラズマチャンバー上部Cellの熱挙動をRCネットワークとして簡略化していますが、特定実機やメーカー仕様を表すものではありません。
