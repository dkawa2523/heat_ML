# 製品化レビュー: 上部Cell温度予測基盤

この基盤は、CAEで作成した上部Cell温度時系列を学習し、ヒーター・ブライン・プラズマ条件から少数計測点の温度時系列を予測するための軽量な製品候補です。

## 設計原則

1. **CAE surrogateを主モデルにする**  
   実測補正は主モデルを置き換えず、機差・センサー差・ゆっくりしたbiasを小さく補正するだけにする。

2. **分割しすぎないが、責務は混ぜない**  
   深いディレクトリ階層は避けつつ、時系列モデル・物理寄りモデル・実測補正は別ファイルに分ける。

3. **モデル比較は共通I/Oで行う**  
   すべての学習モデルは `batch -> standardized ΔT [B, N]` を返す。train/evaluate/predictはモデル種別を意識しない。

4. **少数計測点を前提に過剰な自由度を避ける**  
   3点/4点では `thermal_state_space` と `graph_rc` を主力、TCN/GRU/LSTM/CNNは比較モデルとして扱う。

## コード責務

| ファイル | 役割 |
|---|---|
| `data.py` | CSV読込、条件抽出、ファイル単位split |
| `preprocess.py` | trainのみで標準化fit、範囲情報保存 |
| `control.py` | schedule補間、resampling、effective control |
| `features.py` | 学習windowとbatch作成 |
| `models_sequence.py` | linear/MLP/CNN1D/GRU/LSTM/TCN |
| `models_physics.py` | thermal_state_space / graph_rc |
| `models.py` | モデルregistryとbuild_model |
| `thermal_graph.py` | Cell node/edge表からGraph-RC prior作成 |
| `train.py` | 学習、rollout評価、model_package保存 |
| `predict.py` | forecast / monitor 推論 |
| `correction.py` | 弱い実測補正、residual整理 |
| `residuals.py` | monitor出力から残差解析とoffset artifact作成 |
| `compare.py` | run比較leaderboard |

## モデルファミリー

### A. データ駆動時系列モデル

- `linear_rc`
- `mlp`
- `cnn1d`
- `gru`
- `lstm`
- `tcn`

用途は主に比較・履歴依存の確認です。主力にする場合は、rollout評価とmonitor residualで物理寄りモデルを明確に上回ることを確認してください。

### B. 物理寄りモデル

- `thermal_state_space`: 平衡温度・減衰・残差を分ける安定モデル
- `graph_rc`: Cell位置関係と熱抵抗表をconductance priorとして利用するモデル

少数点・解釈性・CAE surrogateとしての安定性を重視する場合の主力です。

### C. 状態推定・残差補正

- monitor mode: 実測 `T[t]` を使って `T[t+1]` をone-step予測し、残差を監視
- offset補正: センサー別の小さい定常bias
- EMA slow bias: 実測monitor中にゆっくりしたbiasを追従

補正は小さく、補正前後を必ず出力します。補正量が大きい場合はCAE条件、Graph-RC prior、計測点定義を見直してください。

## 推奨ワークフロー

```bash
# 1. base model学習
celltemp train --config configs/config_train.yaml model.name=graph_rc project.run_name=graph_rc

# 2. 実測monitor出力
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc/model_package \
  prediction.mode=monitor \
  prediction.input_table=data/pred/monitor_cases.csv \
  prediction.output_dir=outputs/monitor_graph_rc

# 3. 残差解析とoffset artifact作成
celltemp residuals --config configs/config_pred.yml \
  residuals.input_dir=outputs/monitor_graph_rc \
  residuals.output_dir=outputs/residual_graph_rc

# 4. 弱補正を使ったmonitor確認
celltemp predict --config configs/config_pred.yml \
  model_package.path=outputs/runs/graph_rc/model_package \
  prediction.mode=monitor \
  prediction.input_table=data/pred/monitor_cases.csv \
  prediction.output_dir=outputs/monitor_graph_rc_corrected \
  prediction.correction.enabled=true \
  prediction.correction.mode=offset_ema \
  prediction.correction.artifact=outputs/residual_graph_rc/offset_correction.json
```

## 採用判定

製品利用候補として採用するモデルは、以下を満たすものに限定します。

- test/valのrollout RMSEが安定している
- endpoint error / max errorが許容内
- monitor residualがセンサーごとに説明可能
- Graph-RCの場合、learned/prior conductance比が極端でない
- 補正後だけでなく補正前のbase性能も許容できる
- correction-to-signal ratioが大きすぎない

## 追加開発時のルール

- 新モデルは `models_sequence.py` または `models_physics.py` に追加する。
- すべてのモデルは標準化済み `ΔT [B, N]` を返す。
- 学習・推論のデータ形式をモデルごとに変えない。
- 実測補正は `correction.py` に限定し、主モデルを実測で直接上書き学習しない。
- Neural ODE / PINN / DeepONet / FNO はこの製品範囲では対象外。
