# TopCell-ICP-Thermal Benchmark v0

この例題は、半導体製造装置のICP/RIE系プラズマチャンバー上部Cellを想定した、v6コード評価用の合成CAEデータセットです。

目的は、単に温度を当てることではなく、以下を同じ基盤で評価することです。

- 定数条件CAEからの基本熱応答学習
- 時間変化する `brine / heater / plasma` 入力への時系列予測
- 3点 / 4点など少数計測点での学習・推論
- `thermal_state_space` と `graph_rc` の物理寄りモデル評価
- `cnn1d / gru / lstm / tcn` などデータ駆動モデルとの比較
- Graph-RCの熱抵抗priorの診断
- 実測monitorを模擬した残差解析と弱補正

## 1. 物理想定

上部Cellの代表温度点を以下の4点で表します。

```text
CP, Center, middle, edge
```

簡略熱収支は以下のようなRCネットワークです。

```text
C_i dT_i/dt = Σ_j G_ij (T_j - T_i)
             + plasma入熱
             + heater入熱
             - brine冷却
             + ambient loss
```

生成器では内部的に一次遅れ付きの制御入力を使っています。

```text
brine τ = 4 s
heater τ = 8 s
plasma τ = 1 s
```

v6コードはこの生成式を直接知りません。学習器はCSV時系列とGraph-RCのnode/edge表だけを使います。

## 2. データ構成

### Dataset A: Constant Thermal Response

```text
brine:  10, 20, 30, 40
heater: 80, 120, 160, 200
plasma: 0, 50, 100, 150
initial pattern: cold, nominal, gradient
```

合計:

```text
4 × 4 × 4 × 3 = 192 trajectories
```

### Dataset B: Dynamic Recipe Response

以下の時間変化ケースを含みます。

```text
step_plasma
step_heater
step_brine
recipe
```

代表条件4種類 × schedule 4種類で、合計16 trajectoriesです。

### Dataset D: Pseudo-Measurement Monitoring

実測ログを模擬するため、CAE真値に以下を加えています。

```text
sensor offset
machine bias
slow drift
Gaussian noise
```

このデータは `monitor` と `residuals` の確認用です。

## 3. 主要ファイル

```text
data/raw/                         学習用CAE CSV
configs/cell_nodes.csv             Graph-RCノード表
configs/cell_edges.csv             Graph-RC熱抵抗表
data/pred/conditions.csv           forecast用条件テーブル
data/pred/schedules/*.csv          時間変化schedule
data/pred/monitor_cases.csv        monitor用ケース表
data/pred/monitor_logs/*.csv       疑似実測ログ
scripts/generate_topcell_benchmark.py  データ再生成スクリプト
```

## 4. 推奨評価タスク

### Task 1: 基本補間

```bash
celltemp train --config configs/config_train.yaml \
  model.name=graph_rc \
  project.run_name=graph_rc_topcell_v0
```

### Task 2: 高plasma外挿

```bash
celltemp train --config configs/config_train_holdout_plasma.yaml
```

### Task 3: データ駆動モデル比較

```bash
celltemp train --config configs/config_train.yaml model.name=tcn project.run_name=tcn_topcell_v0
celltemp train --config configs/config_train.yaml model.name=gru project.run_name=gru_topcell_v0
celltemp train --config configs/config_train.yaml model.name=lstm project.run_name=lstm_topcell_v0
```

### Task 4: 3点センサー

```bash
celltemp train --config configs/config_train_3pt.yaml
```

### Task 5: forecast

```bash
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc_topcell_v0/model_package \
  prediction.mode=forecast
```

### Task 6: monitor + residuals

```bash
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc_topcell_v0/model_package \
  prediction.mode=monitor \
  prediction.input_table=data/pred/monitor_cases.csv \
  prediction.output_dir=outputs/monitor_graph_rc

celltemp residuals --config configs/config_pred.yml \
  residuals.input_dir=outputs/monitor_graph_rc \
  residuals.output_dir=outputs/residual_analysis
```

## 5. 評価で見るべきもの

```text
outputs/runs/<run>/metrics/rollout_summary.csv
outputs/runs/<run>/graph/learned_conductance.csv
outputs/runs/<run>/graph/graph_diagnostics.csv
outputs/model_leaderboard.csv
outputs/residual_analysis/residual_summary.csv
outputs/residual_analysis/offset_correction.json
```

## 6. 注意

このデータセットは実装評価用の合成ベンチマークです。物理挙動はRCネットワークとして意味を持つように設計していますが、特定メーカー装置や実機仕様を表すものではありません。
