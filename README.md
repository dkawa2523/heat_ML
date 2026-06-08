# thermal-cell-practical

導体エッチング装置の上部Cellパーツ温度を、CAE過渡応答CSVから学習し、新しい条件で逐次推論するためのシンプルなPython実装です。

設計方針は以下です。

```text
複雑なMLOps・過剰な契約・大量テストは入れない
ただし解析目的に必要な処理は一通り入れる
```

今回の目的では Neural ODE / PINN / DeepONet / FNO は入れていません。少数の計測点温度を固定時間ステップで逐次更新する問題なので、まずは `thermal_state_space` と `graph_rc` を主力にし、`linear_rc / mlp / gru / tcn` を比較用に使います。

## できること

- `temp_<brine>_<heater>_<plasma>.csv` 形式のCAE CSVを読み込み
- CSVファイル単位で train / val / test 分割
- `random / holdout_max / holdout_min / holdout_corner` の簡易split
- trainデータのみで温度、ΔT、操作量を標準化
- 学習dtと推論dtの不一致を検出
- 将来のCAE CSVに時間変化 control 列が追加されても読み込み可能
- ヘッダーあり/なしCAE CSVに対応
- ヒーター・ブラインなどの一次遅れ effective control 特徴量に対応
- 複数モデルを同じデータで比較
  - `linear_rc`
  - `mlp`
  - `gru`
  - `tcn`
  - `thermal_state_space`
  - `graph_rc`
- optional rollout loss
- rollout評価
- 学習曲線、rollout比較、誤差heatmapを出力
- 評価plotは `evaluation.max_plots_per_split` で上限を設定
- `graph_rc` で Cell node/edge 表、熱抵抗表を読み込み
- Graph-RCの表チェック、3点サブグラフ対応、学習後の熱結合CSV/図出力
- Graph-RC conductance prior正則化オプション
- 条件テーブルCSVから複数条件を一括推論
- `schedule_csv` による時間変化する brine/heater/plasma 入力に対応
- schedule補間を `previous` / `linear` から選択可能
- 実測ログから一ステップ先予測残差を見る `monitor` モード
- 実測ログのdtリサンプリング
- 初期温度、制御値範囲、制御値ステップ、長時間rolloutの警告出力
- モデル比較CSVを rollout RMSE 優先で並べ替え

## 主要ファイル

```text
configs/config_train.yaml      学習設定
configs/config_pred.yml        推論設定
configs/cell_nodes.csv         Graph-RCの測定点・入熱/冷却重み
configs/cell_edges.csv         Graph-RCの熱接続・熱抵抗表
src/celltemp/data.py           CSV読込、条件抽出、分割
src/celltemp/preprocess.py     標準化、学習範囲保持
src/celltemp/control.py        schedule補間、実測ログresample、effective control
src/celltemp/features.py       one-step window作成、optional rollout loss用future列
src/celltemp/models.py         各モデル実装
src/celltemp/thermal_graph.py  Graph-RC node/edge表読み込み
src/celltemp/train.py          学習・評価・保存
src/celltemp/evaluate.py       rollout評価
src/celltemp/predict.py        複数条件推論、OOD/安全警告
src/celltemp/compare.py        run比較leaderboard
src/celltemp/plots.py          可視化
```

## インストール

```bash
pip install -e .
```

## 学習

```bash
celltemp train --config configs/config_train.yaml
```

モデルを切り替える場合:

```bash
celltemp train --config configs/config_train.yaml model.name=linear_rc project.run_name=linear_rc
celltemp train --config configs/config_train.yaml model.name=mlp project.run_name=mlp
celltemp train --config configs/config_train.yaml model.name=gru project.run_name=gru
celltemp train --config configs/config_train.yaml model.name=tcn project.run_name=tcn
celltemp train --config configs/config_train.yaml model.name=thermal_state_space project.run_name=thermal_state_space
celltemp train --config configs/config_train.yaml model.name=graph_rc project.run_name=graph_rc
```

rollout loss を軽く入れる場合:

```bash
celltemp train --config configs/config_train.yaml \
  model.name=thermal_state_space \
  train.rollout_loss_weight=0.2 \
  train.rollout_loss_steps=3
```

## 外挿評価split

通常は random split です。

```yaml
split:
  method: random
```

高プラズマ条件をtestに回す場合:

```yaml
split:
  method: holdout_max
  holdout_control: plasma
```

条件空間の高値側cornerをtestに回す場合:

```yaml
split:
  method: holdout_corner
  corner_controls: [brine, heater, plasma]
  corner_direction: max
```


## 時間変化するヒーター・ブラインを使う場合

推論では `schedule_csv` を読みます。設定値がステップ状に変わるレシピでは、線形補間ではなく前値保持が自然です。

```yaml
prediction:
  schedule_interpolation: previous   # previous or linear
```

CAE学習データ側にも時間変化するcontrol列を含められます。ヘッダーありCSVなら例えば以下です。

```text
time,CP,Center,middle,edge,brine,heater,plasma
0,50,52,51,54,20,80,0
1,50.2,52.5,51.8,54.1,20,120,50
```

学習・評価・推論で同じ一次遅れ有効入力を使いたい場合は、以下を有効化します。

```yaml
features:
  use_effective_controls: true
  control_lag_tau:
    brine: 3.0
    heater: 5.0
    plasma: 0.0
```

これはヒーター設定値やブライン設定値が変わっても、実効入熱・実効冷却が即座には変わらないケースの簡易表現です。

## 実測モニタリングモード

通常の `forecast` は初期温度からopen-loopで温度時系列を予測します。実測温度ログを監視する場合は `monitor` を使います。

```bash
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc_demo/model_package \
  prediction.mode=monitor \
  prediction.input_table=data/pred/monitor_cases.csv \
  prediction.output_dir=outputs/predictions_monitor
```

`monitor_cases.csv` の例です。

```text
case_id,dt,log_csv
monitor_case,1.0,data/pred/monitor_logs/monitor_case.csv
```

`log_csv` は以下の列を持ちます。

```text
time,CP,Center,middle,edge,brine,heater,plasma
```

monitorでは、各時刻の実測温度 `T[t]` を状態として使い、`T[t+1]` を一ステップ予測し、実測 `T[t+1]` との差分を出します。open-loopの誤差蓄積を避け、装置状態の残差監視に使えます。

## 3点など少数計測点の場合

3点だけで学習・推論する場合は、`sensor_cols` を変更します。

```bash
celltemp train --config configs/config_train.yaml \
  data.sensor_cols='[Center,middle,edge]' \
  model.graph.ignore_unknown_edges=true \
  model.hidden_dim=16 \
  model.residual_scale=0.02
```

`ignore_unknown_edges=true` にすると、4点用の `cell_edges.csv` に `CP` が含まれていても、3点サブグラフだけでGraph-RCを構築します。少数点では自由度を抑えるため、まずは小さい `hidden_dim` と小さい `residual_scale` を推奨します。

## 推論

```bash
celltemp predict --config configs/config_pred.yml
```

学習済みrunを指定する場合:

```bash
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc/model_package \
  prediction.output_dir=outputs/predictions_graph_rc
```

推論時の `dt` は、デフォルトで学習時の `dt` と一致している必要があります。異なる `dt` を使うと、1秒分のΔTを0.5秒ごとに足すような誤用が起きるためです。

## モデル比較

```bash
celltemp compare --config configs/config_train.yaml
```

出力:

```text
outputs/model_leaderboard.csv
```

`test_rollout_rmse`、なければ `val_rollout_rmse` を優先して並べ替えます。

## 学習結果の出力

```text
outputs/runs/<run_name>/
  resolved_config.yaml
  data_summary.csv
  split.csv
  train_history.csv
  plots/
    loss_curve.png
    val/rollout_*.png
    val/error_*.png
    graph_conductance_prior.png
    graph_conductance_learned.png
  metrics/
    rollout_by_case_val.csv
    rollout_by_sensor_val.csv
    horizon_error_val.csv
    rollout_summary_val.csv
    rollout_by_case_test.csv
    rollout_summary_test.csv
  graph/
    learned_conductance.csv
    graph_diagnostics.csv
    source_weight.csv
  eval_predictions/
    val/*.csv
    test/*.csv
  model_package/
    model.pt
    preprocessor.pkl
    config.yaml
    metadata.json
    graph/cell_nodes.csv
    graph/cell_edges.csv
```

## Graph-RCの表

### `configs/cell_nodes.csv`

```text
sensor,x_mm,y_mm,z_mm,thermal_mass,plasma_weight,heater_weight,brine_weight
CP,0,0,0,1.00,0.70,0.25,0.20
Center,0,0,10,1.20,0.60,0.70,0.25
middle,35,0,10,1.35,0.45,0.60,0.45
edge,70,0,10,1.50,0.30,0.45,0.75
```

### `configs/cell_edges.csv`

```text
src,dst,r_th_K_per_W,contact_type,note
CP,Center,0.80,vertical,plasma side to center
Center,middle,0.55,radial,center to middle ring
middle,edge,0.65,radial,middle ring to edge
Center,edge,1.60,radial_long,weak long-range path
```

内部では、熱抵抗から `G = 1 / R_th` を作り、相対コンダクタンスpriorとして使います。表の `thermal_mass` と `r_th_K_per_W` は正の有限値である必要があります。孤立ノード、重複edge、sensor名不一致は学習前に検出します。

## 推論条件CSV

```text
case_id,t_end,dt,init_CP,init_Center,init_middle,init_edge,brine,heater,plasma,schedule_csv
const_case,15,1,50,52,51,54,20,120,70,
schedule_case,15,1,50,52,51,54,20,100,50,data/pred/schedules/schedule_case.csv
```

`schedule_csv` が空なら定数条件です。指定した場合は、以下のような時系列操作量を読みます。

```text
time,brine,heater,plasma
0,20,80,0
5,20,120,50
10,25,160,90
15,25,120,40
```

推論結果には `prediction_summary.csv` が出力され、範囲外条件や長時間rolloutなどは `warnings/<case_id>.txt` にも保存されます。

## 実装上の最小ルール

1. train/val/testはCSVファイル単位で分割する
2. 全モデルは標準化済み `ΔT` を返す
3. 推論に必要なものは `model_package/` にまとめる

これ以上の厳密な契約や深いディレクトリ階層は入れていません。

## Productized structure note

This package intentionally stays lightweight.  The current product boundary is:

- sequence baselines: `models_sequence.py`
- physics-oriented surrogate models: `models_physics.py`
- model registry: `models.py`
- weak measured-machine correction and residual analysis: `correction.py`, `residuals.py`

The main product workflow is:

```bash
celltemp train --config configs/config_train.yaml model.name=graph_rc
celltemp predict --config configs/config_pred.yml prediction.mode=monitor
celltemp residuals --config configs/config_pred.yml
celltemp predict --config configs/config_pred.yml prediction.correction.enabled=true prediction.correction.mode=offset_ema
```

Residual correction is deliberately weak.  It should explain small machine-to-machine bias, sensor offset, slow drift, and noise.  It must not replace the CAE surrogate.  Always inspect both base and corrected predictions.

See:

- `docs/product_architecture.md`
- `docs/developer_extension_guide.md`
