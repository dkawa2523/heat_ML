# 検証グラフ

公開するベンチマーク検証グラフは、このディレクトリへ集約します。数値の正本は各ベンチマークの
JSON / CSVであり、ここには第三者が波形、真値–予測一致、学習経過、評価境界を確認する図だけを置きます。
通常の`analyze`、`train`、`forecast`、`monitor`が生成する実行単位の図は、それぞれの`work/`出力に残します。

## 評価全体とモデル比較

- [内部／外部評価の境界](00_internal_external_definition.png)
- [外部評価のRMSE比較](01_external_rmse_overview.png)
- [ニューラルモデルの学習・検証履歴](02_training_validation_history.png)
- TopCell外部評価: [時系列](03_topcell_timeseries.png) / [真値–予測散布図](03_topcell_parity.png)
- 線形COMSOL外部評価: [時系列](04_linear_comsol_timeseries.png) / [真値–予測散布図](04_linear_comsol_parity.png)
- 非線形COMSOL外部評価: [時系列](05_nonlinear_comsol_timeseries.png) / [真値–予測散布図](05_nonlinear_comsol_parity.png)
- 高忠実度COMSOL外部評価: [時系列](06_high_fidelity_comsol_timeseries.png) / [真値–予測散布図](06_high_fidelity_comsol_parity.png)
- [既知のmodel-gapに対するRMSE比較](07_model_gap_rmse.png)
- [内部train / validation / testのRMSE](08_internal_rmse_by_split.png)
- TopCell内部test: [時系列](09_topcell_internal_test_timeseries.png) / [真値–予測散布図](09_topcell_internal_test_parity.png)
- 線形COMSOL内部test: [時系列](10_linear_comsol_internal_test_timeseries.png) / [真値–予測散布図](10_linear_comsol_internal_test_parity.png)
- 非線形COMSOL内部test: [時系列](11_nonlinear_comsol_internal_test_timeseries.png) / [真値–予測散布図](11_nonlinear_comsol_internal_test_parity.png)
- [内部testと外部coreの比較](12_internal_test_vs_external_core.png)
- [観測noise感度](13_observation_noise_sensitivity.png)
- [物理RCの未知物理リスク](14_rc_unknown_physics_risk.png)

これら22枚は`benchmarks.neural_comparison.run`が同じsplit、forecast origin、評価条件で再生成します。
元データと集計表は[`../neural_model_comparison/`](../neural_model_comparison/)に保持します。

## 各ベンチマークの直接出力

| 評価 | 通常／core | model-gap |
|---|---|---|
| TopCell | [時系列](topcell_core_prediction_timeseries.png) / [散布図](topcell_core_prediction_parity.png) | [時系列](topcell_model_gap_prediction_timeseries.png) / [散布図](topcell_model_gap_prediction_parity.png) |
| 線形COMSOL | [時系列](linear_comsol_prediction_timeseries.png) / [散布図](linear_comsol_prediction_parity.png) | 対象外 |
| 非線形COMSOL | [時系列](nonlinear_comsol_core_prediction_timeseries.png) / [散布図](nonlinear_comsol_core_prediction_parity.png) | [時系列](nonlinear_comsol_model_gap_prediction_timeseries.png) / [散布図](nonlinear_comsol_model_gap_prediction_parity.png) |
| 高忠実度COMSOL | [時系列](high_fidelity_comsol_prediction_timeseries.png) / [散布図](high_fidelity_comsol_prediction_parity.png) | 放射caseを同じ外部評価内で明示 |

問題の構造、入熱、冷却、計測位置、使用モデルは
[`../benchmark_problem_setups.md`](../benchmark_problem_setups.md)、数値と適用限界は
[`../benchmark_evidence.md`](../benchmark_evidence.md)を参照してください。
