# 高忠実度非線形CAEベンチマーク結果

## 結論

公開の学習・forecast workflowは正常完走し、予測値はfinite、時刻・入力はCAEと一致し、
`truth_*`を変更しても予測が変わらないことを確認しました。workflow判定は
**pass**です。

一方、局所mesh pairに対する流速依存3-node thermal networkのcase平均RMSEは
**0.0956 K**、worstは
**0.1436 K**でした。判定は
**not_resolved_beyond_mesh_difference**です。入口流速は評価境界に保持されていますが、
共通power-law scalar lawを持つboundary conductanceへ結合されています
(`True`)。残差は、固定係数ではなく
この最小流速依存モデルで共役流れをどこまで集約できるかを示します。

## 同定データ内のscreening

| split | n_cases | mean_case_causal_rmse | worst_case_causal_rmse |
|---|---|---|---|
| train | 8 | 0.2146 | 0.3600 |
| val | 1 | 0.3127 | 0.3127 |
| test | 1 | 0.4048 | 0.4048 |

holdout `NT09_cold_high_flow` の因果RMSEは
**0.4048 K**でした。
これはcold/high-flowの組合せ汎化に対する最小thermal networkのscreening evidenceです。
ただし、この10本はglobal-8 meshであり設計精度の絶対誤差判定には使いません。

## ケース別

| case_id | rmse_k | prior_rmse_k | rms_error_over_mesh_difference | max_abs_error_k | max_abs_error_over_mesh_difference | max_hotspot_underprediction_k |
|---|---|---|---|---|---|---|
| HV01_composite_conjugate | 0.0477 | 0.9769 | 0.0962 | 0.1122 | 0.2261 | 0.2375 |
| HV02_composite_radiation | 0.1436 | 0.8009 | 0.2906 | 0.3440 | 0.6988 | 0.0739 |

case RMSEは保守的な隣接mesh差内ですが、最大点誤差は **0.3440 K**、
mesh差に対する最大比は **0.699**でした。したがって平均精度はmesh不確かさから
分離できず、最大点誤差も保守的なmesh差を下回り、現データではモデル差を数値差から分離できません。これは製品合否閾値ではありません。

## センサ別

| case_id | sensor | rmse_k | bias_k | mesh_difference_k | rmse_over_mesh_difference | max_abs_error_over_mesh_difference |
|---|---|---|---|---|---|---|
| HV01_composite_conjugate | chip | 0.0611 | -0.0293 | 0.4962 | 0.1231 | 0.2261 |
| HV01_composite_conjugate | sink_base | 0.0481 | -0.0137 | 0.4954 | 0.0972 | 0.1716 |
| HV01_composite_conjugate | fins | 0.0276 | -0.0023 | 0.4922 | 0.0561 | 0.1123 |
| HV02_composite_radiation | chip | 0.1196 | 0.0813 | 0.4962 | 0.2410 | 0.5399 |
| HV02_composite_radiation | sink_base | 0.1387 | 0.0984 | 0.4954 | 0.2801 | 0.5994 |
| HV02_composite_radiation | fins | 0.1682 | 0.1181 | 0.4922 | 0.3417 | 0.6988 |

## 応答区間別

| case_id | phase | n_timepoints | rmse_k | max_abs_error_k |
|---|---|---|---|---|
| HV01_composite_conjugate | initialization | 3 | 0.0000 | 0.0000 |
| HV01_composite_conjugate | power_excitation | 3 | 0.0372 | 0.0671 |
| HV01_composite_conjugate | airflow_excitation | 2 | 0.0404 | 0.0753 |
| HV01_composite_conjugate | coupled_hot_low_flow | 3 | 0.0765 | 0.1122 |
| HV02_composite_radiation | initialization | 3 | 0.0002 | 0.0002 |
| HV02_composite_radiation | power_excitation | 3 | 0.0533 | 0.0902 |
| HV02_composite_radiation | airflow_excitation | 2 | 0.1136 | 0.1724 |
| HV02_composite_radiation | coupled_hot_low_flow | 3 | 0.2533 | 0.3440 |

区間はleft-ZOHに合わせ、行のcommandが次の区間を駆動した後の応答時刻で集計しています。初期化から
power、airflow、hot/low-flow複合条件へ進むにつれて、モデル誤差が増える箇所を分離しています。

## 放射ペア

| sensor | truth_terminal_radiation_delta_k | predicted_terminal_pair_delta_k | pair_delta_rmse_k | radiation_effect_over_mesh_difference | terminal_predicted_to_truth_delta_ratio_abs |
|---|---|---|---|---|---|
| chip | -0.3813 | -0.0012 | 0.1754 | 0.7685 | 0.0032 |
| sink_base | -0.3832 | -0.0012 | 0.1767 | 0.7735 | 0.0032 |
| fins | -0.3928 | -0.0012 | 0.1838 | 0.7979 | 0.0031 |

放射による最大温度差は **0.3928 K**、隣接mesh差に対する最大比は
**0.798**です。放射項を持たない現行thermal networkで終端まで残った
初期差由来の予測pair差は、
真の終端差の最大 **0.321%**に留まりました。CAE pairから放射の方向性は
確認できますが、mesh収束した
効果量の確定や現行モデルによる放射応答の再現はできません。

## 利用境界

- local-medium meshはbenchmark用途には合格: `True`
- 厳格mesh収束: `False`
- 独立時間刻み収束: `False`
- 同一境界の実験比較実施: `False`
- 同一境界の実験妥当化: `False`

本結果はmodel-form screeningには利用できますが、絶対温度保証、hotspot安全判定、製品設計認証には
利用できません。本結果だけを根拠に空気状態や放射項をcoreへ追加しません。次の基盤評価は、目的対象である
半導体製造装置のwafer/chuck・stage・coolant・process入力を同じ外部評価境界で扱うCAE/実験datasetで
行います。本ケース側では過渡時間刻み収束と実測値による妥当化が未完です。
