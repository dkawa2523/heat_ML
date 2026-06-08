# 開発者・データサイエンティスト向け拡張ガイド

## 新しい時系列モデルを追加する場合

1. `models_sequence.py` に `BaseCellTempModel` を継承したクラスを追加する。
2. `models.py` の `MODELS` に名前を登録する。
3. `forward(batch)` は `standardized ΔT` を `[B, N]` で返す。
4. `celltemp train ... model.name=<new_model>` で既存評価に乗ることを確認する。

## 新しい物理寄りモデルを追加する場合

1. `models_physics.py` に追加する。
2. 物理prior、安定性、補正残差の役割をdocstringに書く。
3. Graph-RCのような診断出力が必要なら、`train.py` の保存処理に最小限追加する。

## 残差補正を拡張する場合

基本順序は以下です。

1. `celltemp residuals` で `residual_long.csv` と `residual_summary.csv` を確認する。
2. offset / EMAで足りるか確認する。
3. 条件依存が明確な場合だけ、Ridge/Huber residualを追加する。
4. 補正量clamp、補正前後出力、補正量レポートを必須にする。

## コードを複雑にしないための禁止事項

- モデルごとに専用DataLoaderを作らない。
- predictの中に新しい学習処理を入れない。
- 補正モデルでCAE surrogateを置き換えない。
- 3点/4点ごとに別コードを作らない。`sensor_cols` とGraph tableで対応する。
- 大量の抽象クラスや深いディレクトリを作らない。
