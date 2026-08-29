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
- `work/data/eval/forecast/`: 学習探索先に含まれない外部forecast 11ケース。
- `work/data/eval/monitor/`: noise、drift、故障、欠測、外乱を持つ外部monitor 5ケース。
- `system.yaml`: truthから意図的にずらしたengineering prior。
- `scripts/definition.py`: generatorとevaluatorが共有する合成truthの唯一の数値定義。

すべて自己完結CSVです。forecastでは通常入力列の将来温度を空欄とし、評価専用の`truth_*`
列を別名で保持します。workflowは`truth_*`を読みません。

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

seed 42、60 epochでの結果です。model gapを除く10ケースの平均RMSEは`0.156 K`、
engineering priorは`7.202 K`でした。

| group | cases | mean RMSE [K] | worst RMSE [K] | 分かること |
|---|---:|---:|---:|---|
| interpolation | 2 | 0.023 | 0.031 | 未学習の中間運転点 |
| extrapolation | 2 | 0.020 | 0.030 | 学習範囲外の高入熱・強冷却 |
| dynamics | 3 | 0.036 | 0.039 | 短周期pulse、未知phase順、複合ramp |
| initialization | 2 | 0.658 | 1.304 | 高温初期状態と初期sensor欠測 |
| sampling | 1 | 0.050 | 0.050 | 可変`dt`をresampleせず積分 |
| model gap | 1 | 9.832 | 9.832 | 温度依存熱損失を線形RCで表せないこと |

最大のcore誤差は、初期4sensor中2点だけを与える`F09`です。これは係数誤差ではなく、未観測
初期温度を一意に復元できない影響を明示します。`F11`の大誤差は失敗ではなく、非線形物理項が
必要な領域を検出するnegative controlです。

### Monitor

| case | 主評価 | 結果 |
|---|---|---:|
| M01 noise only | residual RMSE / filtered truth RMSE | 0.153 / 0.057 K |
| M02 slow drift | raw measurement / filtered truth RMSE | 0.273 / 0.059 K |
| M03 sensor step fault | 最大normalized residual / 検出遅れ | 18.98 / 0 s |
| M04 sensor outages | 欠測率 / filtered truth RMSE | 14.5% / 0.050 K |
| M05 unmodeled heat load | 最大normalized residual / 検出遅れ | 24.33 / 1 s |

M05ではobserverが未知熱源を既知物理として再構成することは期待せず、innovationで速やかに
検出できることを合格条件にしています。絶対sensor offsetは基準温度なしに物理温度と一意分離
できないため、M02は初期bias 0からのdrift追跡を評価します。

### 同定

内部random splitの平均RMSEはtrain `0.023 K`、validation `0.025 K`、test `0.022 K`です。
source/boundary/edgeの最大相対誤差は`1.7%`、actuator tauの最大相対誤差は`7.9%`です。
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
