# TopCell thermal benchmark

このbenchmarkは単に低いRMSEを作るための再生問題ではありません。4温度点、3運転指令、
180秒の過渡系について、次の4点を分離して確認します。

1. 定常条件と単一アクチュエータ励振から物理係数・時定数を同定できるか。
2. 学習に存在しない運転点、複合recipe、初期状態、時間刻みをopen-loop予測できるか。
3. 実測streamのnoise、drift、欠測を扱い、故障と未モデル化熱負荷を検出できるか。
4. 線形RCで表現できない挙動を、低RMSEに見せかけずmodel gapとして露出できるか。

実機精度や特定装置への適合性を証明するものではありません。それらは実CAE・実験データを
同じ評価境界へ入れて確認します。

## データ境界

- `work/data/raw/`: 同定専用228軌道。
  - 定常64運転点 × 3初期温度 = 192軌道。
  - brine/heater/plasmaを個別にstep/ramp励振する36軌道。
- `work/data/eval/forecast/`: 学習探索先に含まれない外部forecast 12ケース。
- `work/data/eval/monitor/`: noise、drift、故障、欠測、外乱を持つ外部monitor 5ケース。
- `system.yaml`: truthから意図的にずらしたengineering prior。
- `scripts/definition.py`: generatorとevaluatorが共有する合成truthの唯一の数値定義。

すべて自己完結CSVです。forecastでは通常入力列に先頭から連続する観測履歴を置き、その後の
将来温度を空欄とします。評価専用の`truth_*`列は別名で保持し、workflowは読みません。

学習run内のrandom train/validation/testはoptimizer選択と回帰確認用です。benchmarkの主結果は、
学習directoryから物理的に分離した`work/data/eval/`に対する評価です。`work/`は生成物なので
Git管理せず、毎回generatorから再構築します。

## 実行

以下はWindows PowerShellの例です。macOS/Linuxでは`py -3`を`python3`へ置き換えます。

```powershell
py -3 benchmarks/topcell/run.py
```

## 基準実行結果

以下はseed 42でのreference resultです。再実行後の判定と完全精度は
`work/outputs/benchmark/benchmark_summary.json`を正とします。

### 外部forecast

seed 42、case-balanced全軌道20 epochでの結果です。model gapを除く11ケースの平均RMSEは
`0.143 K`、engineering priorは`7.185 K`でした。

| group | cases | mean RMSE [K] | worst RMSE [K] | 分かること |
|---|---:|---:|---:|---|
| interpolation | 2 | 0.021 | 0.031 | 未学習の中間運転点 |
| extrapolation | 2 | 0.016 | 0.027 | 学習範囲外の高入熱・強冷却 |
| dynamics | 3 | 0.032 | 0.039 | 短周期pulse、未知phase順、複合ramp |
| initialization | 3 | 0.450 | 1.303 | 高温初期状態、初期sensor欠測、履歴posterior handoff |
| sampling | 1 | 0.050 | 0.050 | 可変`dt`をresampleせず積分 |
| model gap | 1 | 9.823 | 9.823 | 温度依存熱損失を線形RCで表せないこと |

最大のcore誤差は、初期4sensor中2点だけを時刻0で与える`F09`です。これは係数誤差ではなく、
未観測初期温度を一意に復元できない影響を明示します。同じ初期状態・運転・2 sensorを20秒の
因果履歴として与える`F12`は、履歴を評価へ混ぜずorigin以後RMSE `0.038 K`となり、F09の
`1.303 K`から97.1%低減しました。`F11`の大誤差は失敗ではなく、非線形物理項が必要な領域を
検出するnegative controlです。

### Monitor

| case | 主評価 | 結果 |
|---|---|---:|
| M01 noise only | innovation / posterior physical RMSE | 0.167 / 0.063 K |
| M02 slow drift | zero-mean sensor bias / posterior physical RMSE | 0.100 / 0.185 K |
| M03 sensor step fault | 最大NIS / 検出遅れ | 279.17 / 0 s |
| M04 sensor outages | 欠測率 / posterior physical RMSE | 14.5% / 0.077 K |
| M05 unmodeled heat load | peak未知熱推定 / 検出遅れ | 0.758 W / 1 s |

M05ではobserverが既知source空間経路上の未知熱として外乱を推定し、sensor biasへの漏れと分けて
評価します。絶対sensor offsetは基準温度なしに物理温度と一意分離できないため、M02/M03は
共通成分を除いたsensor biasと、gross innovationに対する物理状態保護を評価します。

### 同定

内部random splitの平均RMSEはtrain `0.020 K`、validation `0.023 K`、test `0.019 K`です。
source/boundary/edgeの最大相対誤差は`1.4%`、actuator tauの最大相対誤差は`7.9%`です。
parameter一致は副指標であり、外部trajectory
予測とworst caseを主指標にします。

## 保存結果

```text
work/outputs/runs/topcell/  学習artifactと内部split評価
work/outputs/forecast/      公開forecast workflow出力
work/outputs/monitor/       公開monitor workflow出力
work/outputs/benchmark/
  case_catalog.csv
  forecast_by_case.csv
  forecast_by_group.csv
  monitor_by_case.csv
  parameter_recovery.csv
  benchmark_summary.json
```

各ケースの意図、一次指標、合否境界は
[topcell_benchmark_problem.md](docs/topcell_benchmark_problem.md)に定義しています。
