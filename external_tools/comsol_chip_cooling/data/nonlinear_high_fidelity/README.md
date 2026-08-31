# Nonlinear high-fidelity validation data

このディレクトリは、COMSOL Electronic Chip Cooling原本へ局所meshを適用した外部評価層です。
`celltemp`本体の入力形式は変えず、fin間流路、壁面境界層、chip/contact近傍を解像したCAEと、
実験値を同じ`(case_id, time)`境界へ投入するための表を分離して保持します。

- `mesh/`: mesh構造、skewness品質、局所サイズ、境界層厚さ
- `mesh_convergence*.csv`: 2定常点における隣接mesh差
- `mesh_acceptance.csv`: strict設計基準と用途限定benchmark基準
- `cae_reference.csv`: local-fine定常値とlocal-medium差
- `dynamic/`: local-medium過渡pair、放射差分、実験比較用CAE表
- `experiment/`: 測定値・不確かさを入力するtemplateと試験条件
- `quality_report.md`: 生成結果、利用可能範囲、未完了項目
- `benchmark/`: 本体RCのcase/sensor別評価、放射pair指標、予測時系列、再計算可能なsummary

local-mediumはmodel-form error評価用のbenchmark基準には合格しますが、strict mesh基準には未達です。
設計認証、hotspot安全判定、圧力損失の最終値には使いません。templateの時刻・入力はCAEから生成しますが、
空欄の温度と不確かさは実測値ではありません。対応する測定値がない限り、実験妥当化は未実施です。

`benchmark/` は `benchmark_high_fidelity.py` が生成します。学習artifactはgit管理外の
`work/high_fidelity_benchmark/`に置き、評価側には結果の確認に必要な小さな表だけを残します。

問題設定、局所mesh値、収束結果、実験境界の詳細は
[`../../docs/high_fidelity_validation.md`](../../docs/high_fidelity_validation.md)を参照してください。
