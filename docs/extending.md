# Extending celltemp

変更の種類ごとに所有場所を固定し、装置固有条件をcoreへ持ち込まないための短い案内です。

## 最初に選ぶ境界

| 変更 | 編集場所 | 一緒に確認するもの |
|---|---|---|
| node、sensor、熱経路、入力を変える | projectの`system.yaml` | 単位、sensor mapping、既存artifactとの非互換性 |
| sensorのYAML表現を変える | `io/_sensor_yaml.py` | 単一node、加重平均、未知node、YAML round-trip |
| CAE/実験caseを追加する | 自己完結trajectory CSV | 時刻単調性、左ZOH、case単位split、入力励起 |
| scalar heat lawを追加する | `domain/topology.py`、`engine/laws.py`、`io/system.py` | 正値性、極限、勾配、YAML round-trip |
| 状態方程式の組立を変える | `engine/operators.py` | 保存則、G/R単位、exact解法の適用条件 |
| step・batch積分を変える | `engine/rollout.py`、`engine/integrator.py` | 解析解、可変dt、threshold crossing、gradient |
| forecast/monitorの状態推定を変える | `inference/initialization.py`と対象module | 因果性、欠測、bias gauge、公開re-export |
| 波形・熱設計指標を追加する | `celltemp.analysis` | 単位、欠測、既知波形、適用条件 |
| 学習runの評価表・metadata・保存を変える | `workflows/train_output.py` | causal/conditional定義、既存出力schema、artifact round-trip |
| CAE形式を変換する | `external_tools/...` adapter | raw列と標準trajectoryの対応、truth非参照 |
| 線形COMSOLのforecast/monitor評価を変える | `evaluate_forecast.py` / `evaluate_monitor.py` | 既存7 CSV、summary、screening基準との分離 |
| mesh収束case・QoI・判定を変える | `nonlinear_cases.py` / `run_mesh_convergence.py` | raw再利用境界、隣接mesh差、4 evidence CSVのschema |
| 過渡時間刻み収束を変える | `run_time_step_convergence.py` | 同一schedule・出力時刻、最大BDF刻み、事前固定した8 QoI基準 |
| 非線形COMSOLの生成手順を変える | `run_nonlinear.py` | case/profile選択、raw再利用、role別CSV、QA・放射pair公開 |
| 高忠実度benchmarkの指標・判定を変える | `benchmark_high_fidelity.py` | 同じholdout、3 baseline、mesh差、用途資格 |
| 高忠実度benchmarkの保存・残差図を変える | `benchmark_high_fidelity_output.py` | 既存10成果物、staged directory置換、summary schema |
| 広域非線形benchmarkの指標・判定を変える | `benchmark_nonlinear.py` | 14保持case、3 baseline、残差、放射pair、global-8用途資格 |
| 広域非線形benchmarkの保存・2図を変える | `benchmark_nonlinear_output.py` | 既存23成果物、staged directory置換、summary schema |
| 利用者向け処理を追加する | `workflows`と`cli.py` | 独立した成果物があるか、既存workflowで足りないか |
| 新model familyを試す | 独立した`engine`/`learning` module | 同じsplit、baseline、指標、複数の保持case |

## 典型的な追加

新しい装置はcore classを増やさず、まず`system.yaml`とCSVで表します。

```yaml
actuators:
  - {name: rf_power, tau: 0.8, unit: W, role: heat_input}
  - {name: coolant_temperature, tau: 0.0, learnable: false,
     unit: degC, role: reservoir_temperature}
sources:
  - name: absorbed_plasma_heat
    node_weights: {wafer: 1.0}
    heat_rate: {type: positive_part, control: rf_power, gain: 0.35, threshold: 0.0}
```

`unit`と`role`は表示metadataであり、計算を分岐させません。RF指令と吸収熱を同じ量とみなさず、
`heat_rate`が両者の対応を所有します。装置名による`if`をengineやworkflowへ追加しないでください。

新しい指標はNumPy配列を受け取る純粋関数として`celltemp.analysis`へ置き、CSV探索、設定読込、plot保存は
workflowに残します。benchmark固有の合否値はbenchmark側が所有し、coreへ移しません。
COMSOL adapter間で共有する機械的なleft-limit変換とatomic CSV保存は
`external_tools/comsol_chip_cooling/dataset_support.py`に置きます。列schema、geometry、放射、mesh、role別の
検査はlinear/nonlinearで意味が異なるため、それぞれのdataset moduleに残します。
mesh収束runnerでは、case定義は`nonlinear_cases.py`、COMSOL solveとmesh生成は`run_nonlinear.py`、
QoI差・用途別採否と4つの公開表は`run_mesh_convergence.py`が所有します。単なるprofile追加のために
core modelや共通validatorを変更しません。
非線形caseを追加するときは`nonlinear_cases.py`で入力・role・目的を定義し、標準trajectoryへの変換と
role固有QAは`nonlinear_dataset.py`へ置きます。`run_nonlinear.py`は選択と実行・公開の調停だけを行い、
case固有の物理式や合否値を分岐として追加しません。
高忠実度benchmarkでは、誤差式・baseline・mesh比較・資格判定は`benchmark_high_fidelity.py`、file名・
CSV/JSON書出し・残差図は`benchmark_high_fidelity_output.py`へ置きます。図の都合で評価値を再計算せず、
計算済みのprediction frameとphase表を渡します。
広域非線形benchmarkも、case/group/sensor・残差・放射pairの計算は`benchmark_nonlinear.py`、file名・
CSV/JSON書出し・error matrix・放射pair図は`benchmark_nonlinear_output.py`へ置きます。高忠実度側との
共通化はCSV書出し程度の重複を増やすだけなので行わず、異なる資格と成果物を明示したまま保ちます。
予測modelの比較では`celltemp.analysis.persistence_prediction`、`prediction_error_metrics`、
`prediction_comparison_rows`を使い、benchmarkごとにpersistence生成やRMSE集計を複製しません。
出力は`model_comparison.csv`の1 case × 1 modelへ追加し、modelごとの別schemaや別fileを作りません。
真値peakが記録区間端にあるsensorはpeak時刻評価に適格でないため、温度peak誤差は残して時刻誤差からは
除外します。
thermal impedanceの対象追加はcore編集ではなく、projectの`analysis.thermal_impedance.steps`へ既知の
case、transition、吸収熱変化を追加します。command差を吸収熱へ暗黙変換する装置別分岐は追加しません。

## 追加前の確認

1. 既存のsystem、law、metric、workflow設定のどれに属するか説明できる。
2. 新しいoptionを汎用`extras`辞書へ逃がしていない。
3. train/validation/testは行ではなくcaseまたはrecipe単位で分かれている。
4. 新modelはpersistence、engineering-prior RC、fitted RCと同じ保持caseで比較する。
5. `uv run --locked python quality.py pr`が通り、必要ならTopCell benchmarkも再実行する。

外部entry pointはrepository rootからmoduleとして実行します。たとえば
`python -m benchmarks.topcell.run`と
`python -m external_tools.comsol_chip_cooling.evaluate`です。相対importやshellのcurrent directoryを
暗黙のAPIにしません。
