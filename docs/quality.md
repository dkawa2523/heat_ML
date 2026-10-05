# Quality gates

## 実行

以下はPowerShell、macOS、Linuxで共通の`uv`コマンドです。

```powershell
uv run --locked python quality.py fast
uv run --locked python quality.py architecture
uv run --locked python quality.py pr
```

`fast`はformat、lint、type、unit/property試験を実行します。`architecture`はRuff、Pyrefly、
import-linter、Radonだけを短時間で実行します。`pr`はこれらに全試験とbranch coverageを加えます。
CIはさらにTopCell product benchmarkと`pip-audit .`を実行し、共有Python環境ではなく、このprojectから
解決されるruntime依存だけを監査します。

## 静的検査の役割

| ツール | 判定するもの | 判定しないもの |
|---|---|---|
| [Ruff](https://docs.astral.sh/ruff/settings/) | format、未使用名、危険な記述、import順、局所的な複雑度 | module責務、物理妥当性 |
| [Pyrefly](https://pyrefly.org/en/docs/configuration/) | product APIと外部adapter境界の型、設定値のnarrowing、戻り値の整合 | CAEデータの品質、数値精度 |
| [import-linter](https://import-linter.readthedocs.io/en/stable/contract_types/) | package間の依存方向、leafの純粋性、workflow間の独立性 | 実行時の正しさ、関数内部の複雑度 |
| [Radon](https://radon.readthedocs.io/en/master/commandline.html) | cyclomatic complexityとmaintainabilityの高リスク箇所 | 閾値以下のコードが読みやすいという保証 |

Ruffはcore、tests、TopCellだけでなくneural比較を含む`benchmarks`全体を検査します。Pyreflyへは
quality runnerを実行したPythonのpathを渡し、別OSや別virtual environmentの依存を誤参照しないようにします。
同じ問題を複数ツールで契約化しない。Pyreflyは`src/celltemp`、`tests`、`benchmarks`、
`external_tools`、`quality.py`を一つの型検査範囲とする。外部scriptはpackage importで実行し、pandasから
合否値を取り出す境界では数値型へ明示変換する。型検査のためだけのprotocolや抽象層は追加しない。

```powershell
uv run --locked python -m pyrefly check
uv run --locked python -m radon cc src/celltemp external_tools benchmarks/topcell -s -a
uv run --locked python -m radon mi src/celltemp external_tools benchmarks/topcell -s
```

## 数値検証

| 対象 | 主な試験 |
|---|---|
| command区間のleft/right規約 | `test_domain.py`, `test_io.py` |
| 可変`dt`とaffine解析解 | `test_engine_model.py`, `test_engine_rollout.py`, `test_inference_forecast.py` |
| 対称熱流・energy conservation | `test_engine_model.py`, `test_physical_properties.py` |
| 受動系の上下限 | `test_engine_model.py`, `test_physical_properties.py` |
| actuator解析解・overshootなし | `test_engine_model.py`, `test_engine_rollout.py`, `test_physical_properties.py` |
| 欠測・隠れnode・履歴posterior handoff | `test_engine_observer.py`, `test_inference_forecast.py` |
| zero-mean / reference sensor bias gauge | `test_inference_monitor.py`, `test_workflows.py` |
| forecast境界後の観測拒否 | `test_inference_forecast.py`, `test_workflows.py` |
| gradientとrollout学習 | `test_learning.py` |
| artifact round-trip | `test_artifact.py` |
| 応答・均一性・操作量指標 | `test_analysis.py` |
| power-stepのZth/Rth適格判定 | `test_thermal_impedance.py`, `test_workflows.py` |
| 熱経路G/R/C・連続時間pole | `test_model_analysis.py`, `test_workflows.py` |
| analyze → train → forecast → monitor | `test_workflows.py` |
| trajectory split leakage | `test_dataset.py` |
| NaN/inf予測の拒否と評価点数整合 | `test_analysis.py`, `test_benchmark_acceptance.py`, `test_neural_comparison.py` |
| 長い区間のobserver covariance・PSD・分割合成 | `test_engine_observer.py`, `test_physical_properties.py` |
| float32・加重平均sensorの初期化と勾配 | `test_engine_model.py` |
| tau=0入力・熱流の時刻整合 | `test_engine_rollout.py`, `test_output_integrity.py` |
| 出力名衝突・入力保護・欠測channel解析 | `test_output_integrity.py`, `test_response_availability.py`, `test_workflow_common.py` |
| CAE mesh・時間刻み資格と実solveの来歴照合 | `test_high_fidelity_qualification.py`, `test_time_step_convergence.py` |
| 学習範囲の丸め誤差と実際の外挿の区別 | `test_analysis.py` |
| 図・R²単独利用時の失敗予測拒否 | `test_prediction_figures.py` |
| 重複CSV header・設定の型・保存先・共通読込規約 | `test_io.py`, `test_config.py`, `test_workflow_common.py`, `test_runtime_config.py` |
| 熱回路不要の波形解析・明示stepと観測起点 | `test_analysis_inputs.py`, `test_analysis.py` |
| sensor/node/control・熱収支列の衝突と温度単位 | `test_output_columns.py`, `test_train_output.py` |
| 保存済COMSOL monitorと入力・artifact・条件の照合 | `test_comsol_evaluation.py` |

## Architecture

`.importlinter`は一方向依存と公開workflow間の独立性を検証します。

```text
cli → workflows → analysis/artifact/learning/inference → engine/io → domain/config
```

domainとconfigは独立したleafです。domainはtorch/pandas/YAMLに依存せず、compute層は
pandas/YAMLを読みません。新moduleはexhaustive layersへ追加しない限りarchitecture検査を
通りません。

Radonの自動gateはproduct coreのD–F complexityまたはC maintainabilityだけを失敗にします。C complexityは
設計レビュー対象として記録するが、条件分岐を小関数へ機械的に移すだけの変更は要求しません。外部scriptは
同じ監査で可視化し、用途を説明できない大きな派生物生成器は分割のために残さず削除します。

## Coverage

branchを含む総合coverage下限は`pyproject.toml`にあります。単なる行実行率を上げる試験ではなく、
解析解、不変量、round-trip、失敗入力を優先します。

2026-10-05の責務整理・効率化・入力出力簡素化・COMSOL取り込み版は、WSL Ubuntu / Python 3.12.3 / PyTorch 2.13.0+cpuで
`python quality.py pr`が合格しました。全638試験成功、skipなし、line coverage 93.9%、
branch coverage 80.2%、総合coverage 90.8%で、下限88%を維持しています。
format、Ruff、Pyrefly、import-linter、Radonも合格しています。

今回の回帰試験は、既定analyze→trainの保存先衝突、診断失敗後の基本成果保持、
Torch不要の配列解析・CAE変換、保存済みTopCell出力の採点、参照mesh・物理data・証拠の不一致拒否を含みます。
入力出力の改良では、重複header、誤った型やnull path、dataのCSV規約とruntime明示値、artifactの規約・unit、
名前にstd_やtotalを含むsensor/node/source、曖昧なedge合成名、欠測観測とstep起点、保存済COMSOL monitorの
来歴・設定不一致を確認しました。学習設定の数値検査は重複をまとめ、validator moduleや別の設定schemaを増やしていません。
数値処理ではexact/implicit、可変dt、隠れnode・weighted sensor、threshold crossing、
posterior covariance、parameter/noise変更、予測とgradientの同値性を確認しました。

CPU単一thread、既存quickstartモデル、固定入力100区間、5回の中央値という同じ条件では、
forecastの行列指数計算が101→2回、内部の共通有限性チェックが3,735→1,326回、
forecast時間が92.07→51.27 msになりました。この値は小規模CPU条件の計測であり、GPU・大規模問題の
性能値ではありません。行列値を使う既存cacheを維持し、複雑な更新検知や新しいcache契約は追加していません。
learningはfit内で各trajectoryのTensorを一度準備し、hidden初期状態のprofile軌道を再利用します。

WindowsではPyTorch DLLとcelltemp.exe launcherがapplication controlで拒否されます。
Torchを必要としない配列解析・CAE変換と、`python -m celltemp.cli analyze`はWindowsで成功しました。
学習・予測・監視の実行検証はWSLを使用しています。

入力data、既存の学習epoch、split、受入基準は変更していません。共通dtを運用でも使う変更に伴い、
刻みの異なる既存forecast configへ`dt: null`を明示して、従来の可変刻み動作を維持しています。
新しいCOMSOL solveも実行していません。
quickstartと公開benchmarkは任意診断を有効にするため`project.diagnostics: true`を明示します。
通常設定は既定falseで、診断を生成しなくても基本成果を利用できます。

新しい`config.minimal.yaml`はoverrideなし、既定80 epochsでtrain→forecastが成功しました。
保持test causal RMSEは0.799 K、forecast2 casesは有限で、基本出力とartifactだけが生成されています。
Windowsでもsystemなし・先頭欠測ありのKデータを解析し、通常表と明示step表、K温度軸・K温度差の図を確認しました。

標準設定のquickstartはanalyze→train→forecast→monitorがすべて成功しました。
test causal RMSEは0.894 K、forecast2 casesの温度・標準偏差は有限、熱収支残差は2.7e-15 W以下です。
既定20 epochのTopCellは15/15 checks合格、全8,608評価点のcoverageは1.0、
外部core平均RMSEは0.143 K、worstは1.303 Kでした。fitted forecastとmonitorは保存済みCSVから採点しています。
quickstartの新配置の学習・予測図を目視確認しました。
forecast/monitorはschema 3のCSV・manifestを再生成し、公開consumerから採点しています。
TopCellの設定・system・benchmark定義とCSV計248ファイルは実行前後でSHA-256が一致しました。

既定120 epochの線形COMSOLはtrain→forecast→monitor→evaluateがすべて成功し、11/11 checks合格です。
外部8 casesの平均RMSEは0.036 K、worstは0.083 K、初期化行を除く7,380評価点のcoverageは1.0でした。
monitorの4,515予測点は有限で、欠測132点を維持した区間のRMSEは0.040 Kです。
入力29本とQA・config・systemの計32ファイルは実行前後でSHA-256が一致し、artifactとmanifestの入力来歴も
実際のbytesへ照合しました。数値の正本はCOMSOLの`work/evaluation/summary.json`です。

専門解析の分離後、Windowsで通常`config.yaml`と独立した`analysis.yaml`のanalyzeが成功しました。
通常解析と`work/outputs/impedance/`は別保存先となり、専門解析は従来と同じT01–T05・3 sensorのZthを生成します。
定常Rthは従来どおり未確定として空欄であり、判定閾値や吸収熱量は変更していません。

汎用COMSOL取り込みでは、容量加重温度、alias/weighted sensor、単位、保存指令のright読込、通常IOとの接続、
原本保護と失敗時の既存出力保持を確認しました。起動処理を共有moduleへ移し、既存runnerの公開importを維持しています。
Windows / COMSOL 6.4で既存`electronic_chip_cooling_dataset.mph`をコピーとして読み、唯一の過渡datasetを
自動選択し、3 node・301時刻の`system.yaml`と`trajectory.csv`を生成しました。通常analyzeも成功しています。
既存T03 CSVの領域温度との差は5.0e-9 K以下、right規約で読んだ全300指令区間は参照と一致しました。
元MPHとtemplateのSHA-256は実行前後で一致し、solveと元モデル保存は行っていません。
この確認は保存解の取り込みと通常処理への接続であり、熱結合の自動同定や実機の精度保証を代替しません。
