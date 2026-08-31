# COMSOL Electronic Chip Cooling dataset tool

`celltemp`本体へCOMSOL依存を持ち込まず、Application Libraryモデルのコピーを編集・実行して、
本体と同じself-contained trajectory CSVを作る外部ツールです。用途の異なる2つの問題を分離しています。

- [`docs/problem_definition.md`](docs/problem_definition.md): 固体伝熱 + 線形対流の同定確認用v1
- [`docs/nonlinear_problem_definition.md`](docs/nonlinear_problem_definition.md): 共役熱流動・流速・放射の外部評価用

## 実行

repository rootで次を実行します。

```powershell
uv run python external_tools/comsol_chip_cooling/run.py --overwrite
```

非線形共役熱流動データは次で生成します。

```powershell
uv run --with-editable . python external_tools/comsol_chip_cooling/run_nonlinear.py --overwrite
```

fin間流路、壁境界層、chip/contactを局所細分化した収束評価は次で再生成します。

```powershell
uv run --with-editable . python `
  external_tools/comsol_chip_cooling/run_mesh_convergence.py --reuse-raw
```

局所meshがbenchmark基準を満たした後、情報を集約した2本の過渡trajectoryを生成します。
既定出力では、放射差分、実験比較用CAE表、測定欄だけが空の実験template、品質報告も同時に更新します。

```powershell
uv run --with-editable . python `
  external_tools/comsol_chip_cooling/run_high_fidelity.py `
  --mesh-profile local-medium --overwrite
```

生成済みの局所mesh過渡pairへ本体の流速依存thermal networkを投入するベンチマークは次で実行します。
10本の非線形同定trajectoryから学習し、HV01/HV02を学習から分離したまま公開forecast workflowで
予測します。case/sensor別誤差、未学習prior比較、mesh差との比、放射pair差、truth非参照確認を
`data/nonlinear_high_fidelity/benchmark/`へ保存します。

```powershell
uv run --with-editable . python `
  external_tools/comsol_chip_cooling/benchmark_high_fidelity.py
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
`--reuse-raw` を使えます。単一ケースは `--case T03_power_step_8w` のように指定します。

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
  quality_report.md        品質判定と利用可能範囲
data/nonlinear_high_fidelity/
  mesh/                    global/local meshの構造と品質
  mesh_convergence.csv     2定常点の熱・流動・放射QoI
  mesh_convergence_deltas.csv
  mesh_acceptance.csv      設計基準とbenchmark基準の採否
  cae_reference.csv        最細meshの2定常点と隣接mesh不確かさ
  dynamic/                 benchmark-qualified meshの複合過渡、放射pair、実験比較表
  experiment/              CAEと同一時刻・入力の空欄付き実験templateと受入手順
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

生成後は本体workflowへそのまま渡せます。

```powershell
uv run --with-editable . celltemp train --config external_tools/comsol_chip_cooling/config.yaml
uv run --with-editable . celltemp forecast --config external_tools/comsol_chip_cooling/config.yaml
uv run --with-editable . celltemp monitor --config external_tools/comsol_chip_cooling/config.yaml
uv run --with-editable . python external_tools/comsol_chip_cooling/evaluate.py
```

`evaluate.py`は内部test、8つの外部forecast、5つのcausal monitorをCOMSOL truthへ照合し、
`work/evaluation/`へcase別・sensor別CSVと`summary.json`を保存します。full innovation covarianceの
NIS alert、校正済み`fins`を基準とするsensor bias、欠測区間、未command発熱の物理帰属を別々に
評価し、`truth_*`列を
変えてもforecastが変わらないことも
確認します。判定値は線形COMSOL v1のscreening用であり、実chipの製品許容温度ではありません。

## 現在の基準結果

2026-08-30に全ケースをleft-ZOHで再solveし、case-balanced全軌道学習で10/10判定を通過しました。
内部testのcase平均RMSEは`0.0270 K`、外部forecast 8ケースは平均`0.0363 K`、worst
`0.0830 K`（短pulse）で、未学習priorの平均`7.362 K`を下回りました。3 W未指令発熱は10秒で
検出し、peak推定は`3.182 W`、event中のsensor bias漏れは最大`0.061 K`でした。M02の最終driftは
chip真値`1.5 K`に対して`1.473 K`、sink真値`0.7 K`に対して`0.677 K`、基準finsは厳密に`0 K`です。

これらはvolume-average温度のscreening結果です。最大chip温度との差は最大`0.238 K`、最大温度の
underpredictionは`0.328 K`残るため、hotspot安全判定へ直接使う値ではありません。再実行時の完全な
数値は`work/evaluation/summary.json`を参照します。
