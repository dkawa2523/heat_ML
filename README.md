# thermal-cell-practical

少数の温度センサーと運転指令から、熱系を同定・予測・監視するための汎用的な
集中定数熱基盤です。物理単位のまま動く対称RCモデルを一つだけ持ち、学習、
open-loop forecast、実測monitorが同じ状態方程式を使用します。

## 目的

- 複数のCAEまたは実験過渡データから、熱伝導、熱源、境界熱伝達、アクチュエータ
  応答遅れを同定する。
- 未知運転条件・時間変化レシピに対して、安定した温度軌道を予測する。
- 実測中はKalman observerで内部温度状態とセンサーバイアスを因果的に推定する。
- センサー数と熱状態数を分離し、欠測、隠れノード、可変時間刻みに対応する。

## モデル

ノード温度 `T` に対して、次の物理構造を使用します。

```text
C dT/dt = -L(G) T
          + Σ source_gain · max(actuator - threshold, 0) · source_weights
          + Σ boundary_h · boundary_weights · (boundary_temperature - T)

tau da/dt = command - actuator
measurement = H T + sensor_bias + noise
```

- 各edgeは一つの正のconductanceを共有するため、熱流は常に相反・対称です。
- heat capacity、conductance、source gain、boundary conductance、tauは正値制約を保ちます。
- `exact`積分は行列指数を使い、ゼロ固有値を持つ閉じた熱回路でも逆行列を使いません。
- 可変`dt`を各区間で直接使用します。固定刻みへのresampleは不要です。
- 指令値は明示的なactuator stateを通るため、学習と推論で同じ遅れを使います。

詳細は [product_architecture.md](docs/product_architecture.md) を参照してください。

## 構成

```text
src/celltemp/
  domain/       Trajectory、ThermalSystemSpec
  io/           CSV/DataFrame変換、system YAML読込
  engine/       対称RC、安定積分、Kalman observer
  learning/     軌道分割、multiple-shooting全軌道学習
  workflows/    train、forecast、monitor
  artifact.py   model.pt + system.yaml + metadata.json
  inference.py  ライブラリ用forecast / monitor API
  cli.py        3つの公開コマンド
examples/topcell_quickstart/
  config.yaml   学習・予測・監視で共有する設定
  system.yaml   サンプル熱系
  data/         自己完結CSV
benchmarks/topcell/
  run.py        生成・学習・外部評価を一括実行
  config.yaml   benchmark唯一の設定
  work/         再生成可能な入力と出力（Git管理外）
```

## 入力規約

温度は時刻点、commandは時間区間に属します。

```text
temperature[k] : time[k] での観測
commands[k]    : [time[k], time[k+1]) に適用する指令
```

通常のCSVは時刻ごとに指令を記録するため、既定の`control_convention: left`では
CSVの行`k`を次区間に適用します。内部の`Trajectory.commands`は必ず`N-1`行となり、
`commands[k]`を次の時間区間へ一意に対応させます。

学習対象は`data.directory`と`data.pattern`で自動検出します。追加運用は自己完結CSVを
フォルダへ置くだけです。ファイル名は`case_id`として使いますが、名前から運転条件を解析
しません。

```csv
time,tc_core,tc_shell,heater,coolant
0.0,25.0,25.0,100.0,20.0
1.0,25.8,25.1,100.0,20.0
```

各trajectory CSVは`time + sensors + controls`を持つ自己完結形式です。定数commandも同じ値を
各行へ記録します。CSV単独で再現でき、別の索引との不整合がありません。3つのworkflowは
同じ列規約を用途に応じて次のように使います。

- train: 学習に使うsensor温度を各時刻へ記録する。
- forecast: 先頭行に初期sensor温度、全行に将来commandを記録する。2行目以降の温度は空欄にする。
- monitor: 実測sensor温度と適用commandを各時刻へ記録する。個別の欠測は空欄でよい。

`data.directory`またはruntimeの`input_dir`が処理単位であり、ファイルstemが`case_id`です。
case一覧、予測条件表、schedule参照、log参照は使用しません。forecastは将来の実測値を入力へ
混ぜないよう、2行目以降にsensor値があるCSVを拒否し、初期観測とcommand履歴だけで
open-loop積分します。

通常のrandom splitでは、同一control履歴を持つtrajectoryを自動的に同じsplitへまとめます。
意図的な外挿評価だけ、任意の`case_id,split`表と`split.method: explicit`を使用します。

`dt: null`にすれば可変刻みを許可します。trainで温度欠測を読む場合は
`allow_missing_temperatures: true`を設定します。forecastとmonitorは空欄を自動的に欠測maskとして
扱います。どのworkflowも初期状態を決めるため、先頭行には少なくとも1つのsensor温度が
必要です。

## system.yaml

熱系の構造は一つのYAMLに集約します。

```yaml
nodes:
  - {name: shell, heat_capacity: 2.0}
  - {name: core, heat_capacity: 5.0}
actuators:
  - {name: heater, tau: 3.0}
edges:
  - {nodes: [shell, core], conductance: 0.4}
sources:
  - name: heater_power
    actuator: heater
    node_weights: {core: 1.0}
    gain: 0.2
boundaries:
  - name: ambient
    temperature_intercept: 25.0
    node_weights: {shell: 1.0}
    conductance: 0.05
sensors:
  - {name: tc_core, node: core}
```

`node_weights`はnode名で指定でき、記載しないnodeは0です。sensor名とnode名は異なって
よく、測定されないnodeも状態として保持できます。CSVのsensor列とcontrol列の名前・順序も
この定義から取得するため、`config.yaml`へ重複記載しません。

## 実行

依存関係をインストールします。

```powershell
py -3.13 -m pip install -e ".[dev]"
```

quickstartは1つの設定を3 workflowで共有します。相対パスは常にその設定ファイルのある
ディレクトリから解決され、実行時のカレントディレクトリには依存しません。安全なdirectory
置換のため、各`output_dir`は設定ファイルのあるディレクトリ配下に置きます。

学習:

```powershell
py -3.13 -m celltemp.cli train --config examples/topcell_quickstart/config.yaml
```

予測:

```powershell
py -3.13 -m celltemp.cli forecast --config examples/topcell_quickstart/config.yaml
```

監視:

```powershell
py -3.13 -m celltemp.cli monitor --config examples/topcell_quickstart/config.yaml
```

すべての設定は`key=value`で上書きできます。

```powershell
py -3.13 -m celltemp.cli train --config examples/topcell_quickstart/config.yaml training.epochs=100 training.horizon=90
```

forecast/monitorが読むartifactは、既定では
`project.output_dir/project.run_name/artifact`です。学習runと異なるartifactを使う場合だけ、
top-levelの`artifact`で明示します。

## Python API

CLIとworkflowは同じ公開APIを呼びます。既存のDataFrameから予測する最小構成は次の通りです。

```python
import pandas as pd

from celltemp.artifact import load_artifact
from celltemp.inference import forecast
from celltemp.io import trajectory_from_frame

artifact = load_artifact(
    "examples/topcell_quickstart/work/outputs/runs/thermal_rc_demo/artifact"
)
frame = pd.read_csv("request.csv")
request = trajectory_from_frame(
    case_id="request",
    frame=frame,
    time_col="time",
    sensor_cols=artifact.sensor_names,
    control_cols=artifact.control_names,
)
prediction = forecast(artifact.model, request)
```

入力変換は`celltemp.io`、物理計算は`celltemp.engine`、同定は`celltemp.learning`、学習済み
モデルによる予測・監視は`celltemp.inference`が担当します。

## 学習出力

設定した`project.output_dir/<run_name>/`に次を保存します。quickstartでは
`examples/topcell_quickstart/work/outputs/runs/<run_name>/`です。

```text
artifact/
  model.pt              state_dictのみ。任意コードをpickleしない
  system.yaml           topologyとengineering prior
  metadata.json         fitted physical parameters、範囲、評価値
metrics_by_case.csv     case-balanced train/val/test評価
metrics_by_sensor.csv   センサー別評価
metrics_summary.json    平均・中央値・worst-case
training_history.csv
split.csv
test_predictions/
config.yaml
```

splitは行ではなくtrajectory単位です。同じcontrol履歴で初期温度だけ異なる軌道は分離しません。
学習は短い区間を多数開始点から連続伝播
するmultiple shooting、選択は完全なvalidation軌道RMSEで行います。

## TopCell外部benchmark

学習用228軌道と、学習探索先に含まれない外部forecast 11ケース・monitor 5ケースを分離して
います。ケースの目的、合否条件、最新の基準結果は
[TopCell benchmark](benchmarks/topcell/README.md)に集約しています。

全benchmarkは次の1コマンドで、入力再生成、学習、forecast、monitor、独立評価まで実行します。

```powershell
py -3.13 benchmarks/topcell/run.py
```

## 検証

```powershell
py -3.13 -m pytest -q
py -3.13 quality.py fast
py -3.13 quality.py pr
```

単体試験はエネルギー保存、受動系の上下限、可変刻みsemigroup、actuator解析解、
勾配、欠測observer、artifact round-tripを検証します。integration試験は
`train -> forecast -> monitor`を公開APIで通し、別名sensorから未観測nodeを持つartifactの
forecastと、将来実測を誤って混入した入力を既存出力を壊さず拒否できることも確認します。

## 現時点の境界

- 状態方程式は温度について線形、入力についてthreshold付きaffineです。相変化、放射の
  `T^4`、温度依存物性が主要な系では、物理項を追加する必要があります。
- heat capacityを含む全係数を同時に自由化すると尺度不定になるため、現在はcapacityを
  engineering priorとして固定しています。
- monitorは観測更新に必要なfilter共分散を持ちます。forecastの予測区間は、係数同定の
  uncertaintyを含めて検証できるまでは出力しません。
