# thermal-cell-practical

少数の温度センサーと運転指令から、熱系を同定・予測・監視するための汎用的な
集中定数熱基盤です。物理単位のまま動く熱ネットワークモデルを一つだけ持ち、学習、
open-loop forecast、実測monitorが同じ状態方程式を使用します。

## 目的

- 複数のCAEまたは実験過渡データから、熱伝導、熱源、境界熱伝達、アクチュエータ
  応答遅れを同定する。
- 未知運転条件・時間変化レシピに対して、安定した温度軌道を予測する。
- 実測中はKalman observerで内部温度、既知熱源経路上の未知発熱、識別可能なセンサーバイアスを
  因果的に推定する。
- センサー数と熱状態数を分離し、欠測、隠れノード、可変時間刻みに対応する。

## モデル

ノード温度 `T` に対して、次の物理構造を使用します。

```text
C dT/dt = -L(G_edge(a)) T
          + Σ q_source(a) · source_weights
          + Σ G_boundary(a) · boundary_weights · (reservoir_temperature - T)
          + source_weights^T · unknown_heat

tau da/dt = command - actuator
measurement = H T + sensor_bias + noise
```

- 各edgeは一つの正のconductanceを共有するため、入力依存でも熱流は常に相反・対称です。
- edge conductance、source heat rate、boundary conductanceは同じscalar lawを使います。lawは
  `constant`、しきい値付き`positive_part`、正値・単調な`power_law`の3種類です。
- heat capacity、scalar lawの係数、tauは正値制約を保ちます。特定の冷却方式や現行benchmarkを
  coreの型として持ちません。
- reservoir温度を決めるcontrolとconductanceを決めるcontrolは独立です。
- `exact`積分は温度と一次遅れactuatorを一つの連続系として行列指数で進めます。
  source thresholdを横切る区間は交差時刻で分割し、ゼロ固有値を持つ系でも逆行列を使いません。
- 可変`dt`を各区間で直接使用します。固定刻みへのresampleは不要です。
- 指令値は明示的なactuator stateを通るため、学習と推論で同じ遅れを使います。
- monitorの未知発熱は既存sourceの空間分布を通って温度へ伝播し、sensor biasとは
  別状態として推定されます。sourceがない系だけ各nodeの単位基底を使用します。
- sensor biasは、基準を指定しなければ零平均、`monitor.observer.bias_reference`へ校正済みsensorを
  指定すればそのsensorを0とするgaugeで推定し、出力にもgaugeを明記します。
- gross innovationは検出用のraw NISへ残したまま、Kalman更新では観測noiseを連続的に膨らませ、
  単一sensor faultが物理温度を瞬時に引っ張る影響を抑えます。

詳細は [product_architecture.md](docs/product_architecture.md)、単位とweightの規約は
[units_and_conventions.md](docs/units_and_conventions.md) を参照してください。

## 構成

```text
src/celltemp/
  domain/       Trajectory、ThermalSystemSpec
  io/           CSV/DataFrame変換、system YAML読込
  engine/       熱ネットワーク、安定積分、Kalman observer
  learning/     軌道分割、case-balanced軌道学習
  workflows/    train、forecast、monitor
  artifact.py   model.pt + system.yaml + metadata.json
  inference.py  状態初期化 / forecast / monitor API
  cli.py        3つの公開コマンド
examples/topcell_quickstart/
  config.yaml   学習・予測・監視で共有する設定
  system.yaml   サンプル熱系
  data/         自己完結CSV
benchmarks/topcell/
  run.py        生成・学習・外部評価を一括実行
  config.yaml   benchmark唯一の設定
  work/         再生成可能な入力と出力（Git管理外）
external_tools/comsol_chip_cooling/
  docs/         CAE問題設定と検証境界
  data/         公開可能な正本データと評価結果
  reports/      正本データから再構築する技術レポート入力
docs/           モデル、単位、品質、拡張方針
tests/          単体・property・workflow integration試験
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
- forecast: 先頭から連続する観測履歴と全行のcommandを記録し、履歴後のsensor温度を空欄にする。
  先頭1行だけを観測する従来形は最小の履歴としてそのまま使える。
- monitor: 実測sensor温度と適用commandを各時刻へ記録する。個別の欠測は空欄でよい。

ログ開始時にactuatorが最初のcommandへ整定していない場合だけ、必要なcontrolに
`initial_effective_<control>`列を追加し、先頭行へ開始直前の実効値を記録します。列がなければ
従来どおり最初のcommandを初期値とします。この任意列はtrain、forecast、monitorで共通です。

`data.directory`またはruntimeの`input_dir`が処理単位であり、ファイルstemが`case_id`です。
forecastでは、各履歴行に少なくとも1つのsensor観測を置き、最初の全sensor空欄行以後を
将来区間とします。空欄行より後に観測が再登場するCSVは将来値混入として拒否します。履歴を
因果的にobserverへ通し、最後のposterior node温度とeffective actuatorからopen-loop積分します。
出力はそのforecast originから始まるため、履歴を予測誤差へ混ぜません。

通常のrandom splitでは、同一control履歴を持つtrajectoryを数値丸め誤差を許容して同じsplitへまとめます。
許容値は必要な場合だけ`split.recipe_rtol`と`split.recipe_atol`で変更できます。
意図的な外挿評価だけ、任意の`case_id,split`表と`split.method: explicit`を使用します。

`dt: null`にすれば可変刻みを許可します。trainで温度欠測を読む場合は
`allow_missing_temperatures: true`を設定します。forecastとmonitorは空欄を自動的に欠測maskとして
扱います。どのworkflowも初期状態を決めるため、先頭行には少なくとも1つのsensor温度が
必要です。forecast履歴ではsensor単位の欠測を許しますが、全sensor空欄の行がforecast境界です。
学習trajectoryには、先頭より後にも少なくとも1つの観測が必要です。
学習時に直接観測されないnodeの開始温度は、選択した軌道区間への応答からcaseごとのnuisance
stateとして解析的に推定します。物理係数へ誤った初期温度を吸収させず、artifactへcase固有状態も
持ち込みません。弱観測modeが非現実的な初期温度を取らないよう、観測温度を中心とする
`training.initial_temperature_prior_std`（既定50 K）を使います。出力では、この初期状態推定を
含む`conditional_rmse`と、先頭行の観測だけから積分する`causal_rmse`を明確に分けます。
best modelの選択には、deploymentと同じく将来観測を初期化へ使わない後者を使用します。

## system.yaml

熱系の構造は一つのYAMLに集約します。

```yaml
version: 3
nodes:
  - {name: wafer, heat_capacity: 2.0}
  - {name: chuck, heat_capacity: 5.0}
actuators:
  - {name: heater, tau: 3.0}
  - {name: clamp_pressure, tau: 0.0, learnable: false}
  - {name: coolant_temperature, tau: 0.0, learnable: false}
  - {name: coolant_flow, tau: 0.0, learnable: false}
edges:
  - nodes: [wafer, chuck]
    conductance:
      {type: power_law, control: clamp_pressure, reference: 1.0,
       offset: 0.1, scale: 0.3, exponent: 0.8}
sources:
  - name: heater_power
    node_weights: {chuck: 1.0}
    heat_rate:
      {type: positive_part, control: heater, gain: 0.2, threshold: 0.0}
boundaries:
  - name: coolant
    node_weights: {chuck: 1.0}
    reservoir_temperature:
      {control: coolant_temperature, intercept: 0.0, slope: 1.0}
    conductance:
      type: power_law
      control: coolant_flow
      reference: 1.0
      offset: 0.01
      scale: 0.04
      exponent: 0.8
      offset_learnable: false
      scale_learnable: true
      exponent_learnable: false
sensors:
  - {name: wafer_tc, node: wafer}
```

scalar lawは配置先によってW/KのconductanceまたはWのheat rateになります。固定値は
`{type: constant, value: 0.05}`です。`exact`を使う入力依存conductanceとsourceのpower-law
controlは、区間内一定となる`tau: 0`が必要です。遅れを含む場合は`implicit`が中点の実効入力で
係数を組み立てます。positive-part sourceは一次遅れとthreshold crossingもexact積分できます。
`node_weights`はnode名で指定でき、
記載しないnodeは0です。sensor名とnode名は異なって
よく、測定されないnodeも状態として保持できます。CSVのsensor列とcontrol列の名前・順序も
この定義から取得するため、`config.yaml`へ重複記載しません。
sensorが面積平均や体積平均を表す場合は、単一`node`の代わりに合計1の`node_weights`を指定できます。
一部が直接sensorへ現れない初期状態は、観測行列のnull空間だけを軌道応答から推定します。
`tau`が正のactuatorは`learnable`省略時に学習対象、`tau: 0`は固定の直接入力です。ゼロを
学習priorとして指定することはできません。未知key、文字列化したboolean、空のheat pathは
入力誤記として読込時に拒否します。

## 実行

以下はWindows PowerShellの例です。macOS/Linuxでは`py -3`を`python3`へ置き換えます。
Python 3.10以上の環境へ依存関係をインストールします。再現可能な開発・benchmark環境には
repositoryの`uv.lock`を使用します。

```powershell
uv sync --extra dev --locked
```

通常のPython packageとしては`py -3 -m pip install -e ".[dev]"`でも導入できます。`uv.lock`は
ローカル・CAE benchmarkの固定環境、CIのpip installは宣言した依存範囲と対応Python版の互換性を
検出する役割です。配布名は`thermal-cell-practical`、import packageとCLI名は`celltemp`です。

quickstartは1つの設定を3 workflowで共有します。相対パスは常にその設定ファイルのある
ディレクトリから解決され、実行時のカレントディレクトリには依存しません。相対`output_dir`は
そのproject内に置き、外部storageへ出す場合だけ絶対パスで明示します。完了した結果は既存結果を
backupしてから置換され、処理失敗時は直前の結果を保持します。

学習:

```powershell
py -3 -m celltemp.cli train --config examples/topcell_quickstart/config.yaml
```

予測:

```powershell
py -3 -m celltemp.cli forecast --config examples/topcell_quickstart/config.yaml
```

監視:

```powershell
py -3 -m celltemp.cli monitor --config examples/topcell_quickstart/config.yaml
```

校正済みsensorを絶対biasの基準にする場合だけ、monitor設定へ名前を追加します。そのsensorには
monitorログ内で少なくとも1つの観測が必要です。

```yaml
monitor:
  observer:
    bias_reference: tc_reference
```

forecastの履歴状態推定も、必要な場合だけ同じ場所で測定noiseと初期状態priorを調整できます。
実際に使った値、artifact識別情報、入力ハッシュは出力先の`run_manifest.json`へ保存されます。

```yaml
forecast:
  observer:
    sensor_std: 0.15
    initial_temperature_std: 100.0
    disturbance_process_std: 0.02
```

forecast出力は温度平均に加え、履歴末端の状態共分散と設定したprocess noiseを伝播した標準偏差・
95%区間を持ちます。これは状態・未指令熱の不確かさであり、係数、将来入力、model-formの不確かさは
含みません。区間はGaussian observer仮定に基づく状態区間で、経験的coverageの保証ではありません。
`disturbance_process_std`は保持データの残差に合わせて調整します。`forecast_summary.csv`と
`forecast_coverage.csv`には、将来command、予測sensor温度、時間刻み、予測時間、control slewが
artifactの学習範囲内かも保存され、範囲外caseはCLIにも警告されます。

すべての設定は`key=value`で上書きできます。

```powershell
py -3 -m celltemp.cli train --config examples/topcell_quickstart/config.yaml training.epochs=100 training.horizon=90
```

未知の設定名は`config.yaml`と`system.yaml`の両方でtypoとして拒否されます。

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
    "examples/topcell_quickstart/work/outputs/runs/thermal_network_demo/artifact"
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
  metadata.json         fitted physical parameters、範囲、評価値、構成ファイルhash
metrics_by_case.csv     case-balanced train/val/test評価
metrics_by_sensor.csv   センサー別評価
metrics_summary.json    平均・中央値・worst-case
training_history.csv
split.csv
test_predictions/
config.snapshot.yaml      監査用snapshot。元の基準directoryはartifact metadataに保存
```

splitは行ではなくtrajectory単位です。同じcontrol履歴で初期温度だけ異なる軌道は分離しません。
学習は各epochで全caseを一度ずつ扱い、caseごとの全軌道Huber lossを均等に平均します。
長大ログで計算量を制限するときだけ`training.horizon`へ区間数を指定し、観測可能な開始点から
window rolloutを行います。model選択は完全なvalidation軌道を先頭観測だけから予測したcase平均
`causal_rmse`です。`mean_case_conditional_rmse`も併記し、係数fitと初期化感度を分けて確認できます。

## TopCell外部benchmark

学習用228軌道と、学習探索先に含まれない外部forecast 12ケース・monitor 5ケースを分離して
います。ケースの目的、合否条件、最新の基準結果は
[TopCell benchmark](benchmarks/topcell/README.md)に集約しています。

全benchmarkは次の1コマンドで、入力再生成、学習、forecast、monitor、独立評価まで実行します。

```powershell
py -3 benchmarks/topcell/run.py
```

## 検証

```powershell
py -3 -m pytest -q
py -3 quality.py fast
py -3 quality.py pr
```

単体試験はエネルギー保存、受動系の上下限、可変刻みsemigroup、actuator解析解、
勾配、欠測observer、artifact round-tripを検証します。integration試験は
`train -> forecast -> monitor`を公開APIで通し、別名sensorから未観測nodeを持つartifactの
forecast、観測履歴からのhidden state推定、およびforecast境界後の実測を混入した入力を
既存出力を壊さず拒否できることも確認します。

## 現時点の境界

- 状態方程式は温度について線形で、区間ごとの入力から正値の係数を組み立てます。edge、source、
  boundaryは共通scalar lawを使います。相変化、放射の`T^4`、
  温度依存物性が主要な系では、温度依存の物理項を追加する必要があります。
- heat capacityを含む全係数を同時に自由化すると尺度不定になるため、現在はcapacityを
  engineering priorとして固定しています。
- forecastの95%区間は状態推定と未指令熱process noiseだけを伝播します。係数、将来入力、放射などの
  model-form uncertaintyは含まず、経験的に校正済みの予測区間でもありません。
  `run_manifest.json`にも区間の意味を明記します。予測温度が学習温度域を外れた場合は、入力範囲内でも
  coverage警告を出します。
- 外部基準なしでは全sensor共通offsetと一様な物理温度ずれを分離できません。この場合の
  `sensor_bias`は零平均です。校正済みsensorがある場合だけ`bias_reference`へ名前を指定し、
  そのsensorのbiasを0として他sensorの絶対offsetを推定できます。
