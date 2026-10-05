# COMSOL Electronic Chip Cooling dataset tool

`celltemp`本体へCOMSOL依存を持ち込まず、Application Libraryモデルのコピーを編集・実行して、
本体と同じself-contained trajectory CSVを作る外部ツールです。用途の異なる2つの問題を分離しています。

既存の解析済み`.mph`から熱回路下書きとCSVを取り込む場合は、
[汎用COMSOL取り込み](../comsol/README.md)を使います。以下のgeneratorは用途別CAEの生成・検証を所有します。

- [`docs/problem_definition.md`](docs/problem_definition.md): 固体伝熱 + 線形対流の同定確認用v1
- [`docs/nonlinear_problem_definition.md`](docs/nonlinear_problem_definition.md): 共役熱流動・流速・放射の外部評価用
- [`../../docs/benchmark_problem_setups.md`](../../docs/benchmark_problem_setups.md): 入熱・冷却・計測領域・評価モデル・`.mph`系譜の図

本体の入力CSVはtime・sensor・control、予測/監視の保存CSVは
`sensor.<名前>.<量>`、`node.<名前>.<量>`、`control.<名前>.<量>`です。外部評価は真値との比較と
用途別合否を所有し、熱状態の計算は本体へ委譲します。monitor評価では既存manifestの入力・artifact・
解決済条件を入口で照合し、古い別runを現在の評価へ混ぜません。旧保存CSVはforecast/monitorを再実行してください。
`forecast.dt: null`は学習と異なる刻みの公開評価CSVをそのまま読むための明示設定です。

## 実行

repository rootで次を実行します。

```powershell
uv run --locked --with-editable . python -m external_tools.comsol_chip_cooling.run --overwrite
```

非線形共役熱流動データは次で生成します。

```powershell
uv run --locked --with-editable . python -m external_tools.comsol_chip_cooling.run_nonlinear --overwrite
```

fin間流路、壁境界層、chip/contactを局所細分化した収束評価は次で再生成します。

```powershell
uv run --locked --with-editable . python -m `
  external_tools.comsol_chip_cooling.run_mesh_convergence --reuse-raw
```

局所meshがbenchmark基準を満たした後、情報を集約した2本の過渡trajectoryを生成します。
既定出力では、放射差分、実験比較用CAE表、測定欄だけが空の実験template、品質報告も同時に更新します。

```powershell
uv run --locked --with-editable . python -m external_tools.comsol_chip_cooling.run_high_fidelity `
  --mesh-profile local-medium --overwrite
```

同じHV02過渡入力・出力時刻を維持したまま、COMSOL BDF solverの最大内部刻みだけを2 sから1 sへ
細分化する時間刻み収束は次で実行します。これは2本の新しいCOMSOL solveを行います。既存rawが両方
揃う再集計時だけ`--reuse-raw`を指定します。

```powershell
uv run --locked --with-editable . python -m `
  external_tools.comsol_chip_cooling.run_time_step_convergence
```

判定は結果を見る前に、領域平均温度0.05 K、最高温度0.10 K、圧力差・放射熱量0.5%へ固定しています。
結果の正本は`time_step_convergence.csv`、各隣接pairの履歴と比較trajectoryは`temporal/`へ保存します。
`dynamic/cae_reference.csv`の時間資格は、合格pairのfine側と実際の参照データのcase・mesh・最大内部刻み・
データ署名が一致し、rawとsolver設定のSHA-256来歴を検証できる場合だけ付与します。別の刻みやadaptive
solverの参照データへ資格と時間不確かさを転写しません。来歴のない既存rawはscreening用途を維持します。最初のpairが
不合格なら受入値を変えず、既存のfine側をcoarse側として次の細分刻みを追加します。

現在の2 s / 1 s比較は平均温度3項目と放射熱量が基準を超え、時間離散化は未資格です。0.5 sへの追加
細分化は後日実施し、それまではscreening用途を維持します。

生成済みの局所mesh過渡pairへ本体の流速依存thermal networkを投入するベンチマークは次で実行します。
10本の非線形同定trajectoryから学習し、HV01/HV02を学習から分離したまま公開forecast workflowで
予測します。case/sensor別誤差、persistence・未学習prior・fitted RC比較、mesh差との比、放射pair差、truth非参照確認を
`data/nonlinear_high_fidelity/benchmark/`へ保存します。3 baselineのcase別worst sensor・peak値・適格な
peak時刻は`model_comparison.csv`が正本です。時間・温度・入力・圧力損失・放射熱量に対する
記述的な残差依存は`residual_dependence.csv`へ保存します。protocol区間に沿ったworst外部caseの波形は
`docs/validation_figures/high_fidelity_comsol_prediction_timeseries.png`、全外部点の真値–予測散布図とR²は
`docs/validation_figures/high_fidelity_comsol_prediction_parity.png`です。相関は因果や追加物理の採用判定と
しては扱いません。

```powershell
uv run --locked --with-editable . python -m `
  external_tools.comsol_chip_cooling.benchmark_high_fidelity
```

global-8の外部14ケース（内外挿、短pulse、hot start、放射pair）を同じモデルで広くscreeningする場合は
次を実行します。結果は`data/nonlinear/benchmark/`のCSV/JSONと、非放射・放射model gapを分けた
真値–予測時系列図およびparity plotへ保存し、公開図は`docs/validation_figures/`へ集約します。外部report
builderは使いません。case/sensor表と予測時系列には
persistence、engineering prior、fitted RCを並べ、case別の
worst sensor・peak比較は`model_comparison.csv`へ一本化します。

```powershell
uv run --locked --with-editable . python -m `
  external_tools.comsol_chip_cooling.benchmark_nonlinear
```

直前の高忠実度benchmarkと全く同じartifactを使う場合だけ、再学習せず明示的に渡せます。

```powershell
uv run --locked --with-editable . python -m `
  external_tools.comsol_chip_cooling.benchmark_nonlinear `
  --artifact external_tools/comsol_chip_cooling/work/high_fidelity_benchmark/runs/nonlinear_high_fidelity_network/artifact
```

入口流速は共通scalar lawを使うboundary conductanceへ結合されます。case別補正はなく、残る誤差は
固体3-nodeへの集約、空気側熱状態、空間分布、放射、データ忠実度を含むmodel-form evidenceとして
評価結果へ残します。

既定はmesh level 8です。COMSOL raw tableが残る再開時は `--reuse-raw`、代表ケースだけなら
`--case NT01_power_levels`、mesh比較なら `--mesh-size 7` のように指定します。

runnerは次を自動で行います。

1. Heat TransferライセンスをcheckoutできるCOMSOL 6.4 installationを選ぶ。
2. `comsol/RunChipCoolingCase.java`をCOMSOL compilerでcompileする。
3. 原本を`loadCopy`し、各scheduleの過渡studyをsolveする。
4. COMSOL text tableを温度degC・入力物理単位のCSVへ変換する。
5. forecast/monitor用の観測maskと評価専用truthを作り、全fileをQAする。

`run_nonlinear.py` はこれに加えて、時刻0の定常共役熱流動を過渡初期場へ引き継ぎ、流速command、
温度依存air、任意の表面間放射を解きます。通常ケースは10秒、短pulseだけ1秒間隔です。

installationを明示する場合は `--comsol-root`、COMSOLのraw結果から変換だけを再開する場合は
`--reuse-raw` を使えます。選択した全raw tableが揃っていれば、この再変換にはCOMSOL installationや
license checkoutは不要です。不足するtableが1つでもあれば通常どおりCOMSOLを選択します。単一ケースは
`--case T03_power_step_8w` のように指定します。solve中は既存rawを保持し、生成物の構文と入力一致を
確認してから置換します。

## 出力

```text
data/
  train/                 16 identification trajectories
  eval/forecast/          8 external open-loop cases
  eval/monitor/           5 causal monitoring cases
  qa_summary.csv          generated data QA summary (実行には不要)
work/                     git管理外
  schedules/              left-ZOHで適用するCOMSOL入力
  raw/                    raw COMSOL tables
  logs/                   one solver log per physical case
  models/
    electronic_chip_cooling_dataset.mph
```

非線形データは既存v1を上書きせず、次へ出力します。

```text
data/nonlinear/
  train/                  10 conjugate-flow identification cases
  eval/forecast/           9 external open-loop cases
  eval/monitor/            3 physical-disturbance cases
  eval/model_gap/          5 paired radiation cases
  qa_summary.csv           generated QA evidence
  radiation_pairs.csv      全ケース生成時に更新する放射あり/なしのpaired差分
  benchmark/               独立14ケースの誤差・残差・放射pair評価
data/nonlinear_high_fidelity/
  mesh/                    global/local meshの構造と品質
  mesh_convergence.csv     2定常点の熱・流動・放射QoI
  mesh_convergence_deltas.csv
  mesh_acceptance.csv      設計基準とbenchmark基準の採否
  cae_reference.csv        最細meshの2定常点と隣接mesh不確かさ
  dynamic/                 benchmark-qualified meshの複合過渡、放射pair、実験比較表
  temporal/                BDF最大刻み2 s / 1 sの同一境界trajectory
  time_step_convergence.csv
  benchmark/               局所mesh 2ケースの誤差・残差・mesh差比較
  experiment/              CAEと同一時刻・入力の空欄付き実験templateと受入手順
  quality_summary.json     用途境界と主要な品質判定
```

非線形CSVには従来の6基本列に `inlet_air_velocity` を加え、最高温度、outlet温度、圧力差、放射熱量、
熱収支、非公開外乱、実効流速を評価専用列として保持します。mesh level 8はmesh independentではないため、
本データの用途はモデル評価・screeningであり、設計認証値ではありません。

高忠実度層の問題設定、局所サイズ、境界層、収束条件、実験同一境界は
[`docs/high_fidelity_validation.md`](docs/high_fidelity_validation.md)にまとめています。実測値が未提供の場合、
`experiment/experiment_template.csv`を実データとして扱わず、妥当化状態を未実施のまま保持します。

`data`の各CSVはそれだけで読め、manifest依存はありません。`work/models`の`.mph`は、名目8 W
ケースで編集・solve済みの確認用コピーです。Application Library原本は変更されません。
Application Library更新時のdomain再確認には、補助class `comsol/InspectGeometry.java` がvolumeと
centroidを表示します。通常のdataset生成には使いません。

現在のrunnerはCOMSOL installationの
`applications/Heat_Transfer_Module/Tutorials,_Forced_and_Natural_Convection/chip_cooling.mph`を
`ModelUtil.loadCopy`で開きます。線形の代表コピーは`work/models/electronic_chip_cooling_dataset.mph`、
global-8非線形の代表コピーは`work/nonlinear/mesh_8/models/`の
`electronic_chip_cooling_conjugate.mph`と`electronic_chip_cooling_radiation.mph`です。
high-fidelity local-mediumは`.mph`を保存せず、`work/nonlinear/mesh_local_medium/raw/`と
`logs/`へsolver結果とlogを保持します。正確な実ファイル名と系譜は上記problem setup figuresを参照して
ください。

生成後は本体workflowへそのまま渡せます。

```powershell
uv run --locked --with-editable . celltemp analyze --config external_tools/comsol_chip_cooling/config.yaml
uv run --with-editable . celltemp train --config external_tools/comsol_chip_cooling/config.yaml
uv run --with-editable . celltemp forecast --config external_tools/comsol_chip_cooling/config.yaml
uv run --with-editable . celltemp monitor --config external_tools/comsol_chip_cooling/config.yaml
uv run --locked --with-editable . python -m external_tools.comsol_chip_cooling.evaluate
```

通常の`config.yaml`は波形解析・学習・予測・監視の入口です。T01–T05の過渡Zthは、必要な場合だけ
独立した[`analysis.yaml`](analysis.yaml)を使い、`work/outputs/impedance/`へ保存します。

```powershell
uv run --locked --with-editable . celltemp analyze --config external_tools/comsol_chip_cooling/analysis.yaml
```

この公開trajectoryは600 s時点でも
終端Zth勾配が設定した定常判定を満たさないため、定常Rthは未確定として空欄になります。閾値を緩めて
最終値を定常値へ読み替えず、必要なら試験時間を延長した独立caseを生成してください。

`evaluate.py`は内部test、8つの外部forecast、5つのcausal monitorをCOMSOL truthへ照合し、
`work/evaluation/`へcase別・sensor別CSVと`summary.json`を保存します。full innovation covarianceの
NIS alert、校正済み`fins`を基準とするsensor bias、欠測区間、未command発熱の物理帰属を別々に
評価し、`truth_*`列を
変えてもforecastが変わらないことも
確認します。`docs/validation_figures/linear_comsol_prediction_timeseries.png`と
`docs/validation_figures/linear_comsol_prediction_parity.png`は外部8 caseの真値・予測を波形とR²付き散布図で
示します。判定値は線形COMSOL v1のscreening用であり、実chipの製品許容温度ではありません。

## 現在の基準結果

2026-08-30に全ケースをleft-ZOHで再solveし、case-balanced全軌道学習で11/11判定を通過しました。
内部testのcase平均RMSEは`0.0270 K`、外部forecast 8ケースは平均`0.0363 K`、worst
`0.0830 K`（短pulse）で、未学習priorの平均`7.362 K`とpersistenceの平均`17.496 K`を全ケースで
下回りました。3 W未指令発熱は10秒で
検出し、peak推定は`3.182 W`、event中のsensor bias漏れは最大`0.061 K`でした。M02の最終driftは
chip真値`1.5 K`に対して`1.473 K`、sink真値`0.7 K`に対して`0.677 K`、基準finsは厳密に`0 K`です。

これらはvolume-average温度のscreening結果です。最大chip温度との差は最大`0.238 K`、最大温度の
underpredictionは`0.328 K`残るため、hotspot安全判定へ直接使う値ではありません。再実行時の完全な
数値は`work/evaluation/summary.json`を参照します。
