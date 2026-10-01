# 熱時系列解析基盤の完成計画

## 1. 完成させるもの

本基盤の目的は、CAEまたは実験の温度時系列と、時間変化する入熱・冷却条件を受け取り、次を一貫して行う
ことである。

1. 生の熱時系列から、peak、昇温・冷却速度、応答時間、整定、温度均一性、適格なZth/Rthを計算する。
2. 解釈可能な集中定数熱回路を同定し、熱容量、熱経路、時定数、保持caseでの誤差を示す。
3. 未知recipeの入熱・冷却波形から部品温度をopen-loop予測し、温度波形、不確かさ、適用範囲、熱収支を示す。
4. 必要な設備だけ、同じ熱回路へ観測を逐次同化して未知発熱・sensor bias・欠測をmonitorする。

中心となる利用順序は `analyze -> train -> forecast` である。`monitor`は任意機能であり、基盤完成のために
診断機能を増やすことはしない。

### 1.1 対象範囲

本基盤が扱うのは、対象部品へ吸収された熱量またはそれに対応する既知commandと、冷却境界の時間変化から、
部品温度を解析・予測する問題である。plasma/RFの電磁場、反応、吸収率そのものは計算しない。plasma由来の
入熱は、測定、校正、または外部解析で得た `absorbed_heat_w` 等の入力として受け取る。

含めるもの:

- 複数nodeの伝導、source、冷却境界、入力依存係数、actuator遅れ
- 可変時間刻み、欠測、隠れnode、複数sensor
- 時間変化する発熱、冷却温度、flow・pressure等の熱伝達command
- 因果的なmulti-step予測、状態不確かさ、熱収支
- 同じ保持case上でのengineering prior、persistence、fitted RC比較

含めないもの:

- plasma、CFD、電磁場solver
- 外部report builder、HTML/PPT生成基盤
- 根拠dataなしのradiation law、chamber memory node、model family追加
- 本モデル単独による安全保護、製品合否、hotspot保証

## 2. 完成の定義

「software基盤の完成」と「対象装置での妥当化完了」を分ける。前者はrepository内で完成できるが、後者には
対象装置のdataと用途別基準が必要である。

### 2.1 Software基盤 v1 の完了条件

- 入力は自己完結trajectory CSV、`system.yaml`、`config.yaml`の3種類だけである。
- 新規利用者がREADMEの3コマンドでanalyze、train、forecastを再現できる。
- analyzeはcase/sensor/control指標と、温度・入力・uniformityの波形図を返す。
- trainは保持caseについてfitted RC、engineering prior、persistenceを同じ表と図で比較する。
- trainはthermal path、mode、mean/worst error、peak値・時刻誤差、適用範囲を返す。
- forecastは温度、95%状態区間、入力、sensor spanを一つのcase図で確認できる。
- forecastはpeak、応答、uniformity、入力積分、適用範囲、source/boundary/storage熱収支を返す。
- 主要結果を最初に読むfileがREADMEから一意に分かり、詳細診断は主結果と区別される。
- quickstartと独立benchmarkが再現でき、Ruff、Pyrefly、import-linter、Radon、全試験が通る。
- 使用されない旧処理、重複文書、派生reportは残さない。

### 2.2 対象装置での妥当化完了条件

- CAEのmesh収束と時間刻み収束が、事前に定めた用途別基準を満たす。
- 実験とCAEが同一のcase、時刻、入力、観測量で比較され、sensor配置・時定数・校正不確かさが記録される。
- 独立recipeでmeanだけでなくworst error、peak、peak時刻、hotspot margin、外挿条件を満たす。
- 許容温度、uniformity、予測horizon、更新triggerを利用者が定め、model cardへ固定する。
- 安全保護は本モデルから独立している。

実測値と用途別閾値がない現状では、2.2を完了扱いにしない。空template、CAE値、任意の閾値で代用しない。

## 3. 現在地

| 領域 | 状態 | 完成までの残り |
|---|---|---|
| domain / engine / IO | 完了 | 物理不変条件を維持し、機能要求なしに分割しない |
| `celltemp analyze` | 完了 | summary、case表、case図の読み順を通常利用経路へ固定済み |
| `celltemp train` | 完了 | 保持caseの3 baseline比較とvalidation overviewを通常runへ追加済み |
| `celltemp forecast` | 完了 | 主要予測図と熱収支図を通常runへ追加済み |
| `celltemp monitor` | 任意機能として完了 | 実機dataなしにQ/Rやalarmを増やさない |
| TopCell / COMSOL benchmark | screeningとして完了 | 既存evidenceを維持し、装置資格と混同しない |
| RC–ニューラル比較 | screening完了、product不採用 | 複数seedやresidual候補は独立dataが必要な場合だけ再検討 |
| 観測noise / 未知物理 | 推論時感度とmodel-gapを分離してscreening完了 | noisy学習・parameter uncertaintyは実測要件が定まるまで増やさない |
| repository cleanup / v1境界 | 完了 | 参照のない派生物を戻さず、公開7層を維持する |
| high-fidelity CAE | model-form screeningのみ | 2 s / 1 s時間刻みは未合格、次の細分化と実験比較は後日実施 |
| chamber実機資格 | 未着手 | 吸収熱・冷却・温度の実測dataと事前承認基準が必要 |

Step 1–4が完了し、2.1で定義したsoftware基盤v1は完成した。Step 5のCAE・実測資格は未完了である。
Step 6はproductへmodelを増やさない評価専用screeningとして実施し、RCを置換する証拠がないことを確認した。
独立実測dataが揃うまではproduct modelを追加せず、screening結果を実機精度保証として扱わない。

直近までの責務分離で、coreの循環依存、大規模module、外部report builderは整理できた。これ以上、Radon値や
file行数だけを理由に小さなrunnerやvalidatorを分割しても、利用者価値は増えない。したがって、予定していた
線形`run.py`の追加分割は中止し、以後は下記の完成手順だけを優先する。

## 4. 完成までの実装順序

### Step 1 — 学習結果を判断可能にする（P0、完了）

対象: `src/celltemp/workflows/train_output.py`

- validation/testの各caseを、`fitted_rc`、`engineering_prior_rc`、`persistence`で同じ因果条件から予測する。
- `model_comparison.csv`を1 case × 1 modelで出力する。
- RMSE、MAE、最大誤差、worst sensor、peak温度誤差、適格なpeak時刻誤差を共通列で保存する。
- `test_prediction_timeseries.png`にworst test caseの全sensor波形、`test_prediction_parity.png`に
  全test点の真値–予測散布図とR²を示す。model別指標はCSVへ残し、図を指標一覧にしない。
- conditional fitは学習診断に残すが、model採否は先頭観測からのcausal open-loopを使う。

受入条件:

- split、学習係数、artifact schema、CLI optionを変えない。
- quickstartの保持caseで3 baselineが同じ時刻・同じ観測境界を使う。
- 既存TopCell/COMSOL評価値を変えず、通常trainでもmodelの有用性を判断できる。
- 新しいmodel class、設定option、汎用report層を追加しない。

実装結果:

- quickstartのvalidation 5 case・test 6 caseを各3 model、計33行で比較した。
- test平均causal RMSEはfitted RC 0.893532 K、engineering prior RC 3.660352 K、persistence 2.080840 Kだった。
- 変更前から存在する成果物は実行時刻を持つartifact metadata以外すべてSHA-256が一致した。
- split、学習係数、artifact schema、CLI・configを変更していない。
- TopCell外部benchmarkは15/15、線形COMSOL screeningは11/11の既存判定を維持した。
- 線形COMSOLの外部forecastは平均0.036276 K、worst 0.082974 Kで、prior平均7.361789 Kと
  persistence平均17.495760 Kを下回った。

### Step 2 — 予測結果の主要図を完成させる（P0、完了）

対象: `src/celltemp/workflows/forecast_output.py`

- caseごとに、予測sensor温度と95%状態区間を同じpanelへ描く。
- 同じ図に時間変化する入熱・冷却commandとsensor spanを置く。
- 将来truthはforecast入力に存在しないため、通常予測図へtruthや誤差を捏造しない。
- 既存`energy_balance.csv`と熱収支図は、物理解釈用の第二図として維持する。

受入条件:

- quickstartの定常入力caseと複合recipe caseの両方で図が生成される。
- CSVの数値を図側で再計算せず、既存forecast結果を表示する。
- out-of-range警告、予測CSV、指標、熱収支の値が変更前後で一致する。

実装結果:

- quickstartの定常入力caseと複合recipe caseへ`figures/forecast_<case_id>.png`を生成した。
- 図は保存対象と同じforecast DataFrameの温度、上下95%区間、command、sensor spanだけを表示する。
- 変更前の予測CSV、全指標、coverage、熱収支CSV・図の10成果物はSHA-256が一致した。
- out-of-range警告は変更前後で一致し、manifestは実行時刻以外のschema・設定を変えていない。
- config、CLI、artifact、解析指標、model計算、外部report層を追加していない。

### Step 3 — 最小利用経路と出力を確定する（P0、完了）

対象: root README、quickstart、workflow integration試験

- cleanな一時出力先で `analyze -> train -> forecast` を順に実行する。
- README冒頭に、各処理で最初に読む結果を次のように固定する。
  - analyze: `summary.json`、`case_metrics.csv`、case波形図
  - train: `metrics_summary.json`、`model_comparison.csv`、validation図、`thermal_paths.csv`
  - forecast: `forecast_summary.csv`、予測図、`energy_balance.csv`
- split、history、sensor別表などは再現・詳細確認用であることを明記する。
- quickstartが時間変化する発熱・冷却を含むことを確認する。これはsolver妥当化ではなく操作例と明記する。

受入条件:

- 初見の利用者が外部COMSOL toolやmonitorを読まずに主経路を完走できる。
- core利用にCOMSOL、外部report、notebookを要求しない。
- 同じ情報を持つfileが複数ある場合は正本を一つにし、参照のない派生物を削除する。

実装結果:

- cleanなquickstartコピーで`analyze -> train -> forecast`を順に完走した。
- analysis 36 case・36図、train 25/5/6 case split・33 baseline比較行、forecast 2 case・主要図2枚を確認した。
- README冒頭へ3コマンドと最初に読む成果物を固定し、後段に重複していた基本コマンド列を削除した。
- workflow統合試験も基本順序で実行し、analysis、train、forecastの主要成果物を検証する形へ揃えた。
- core出力を粒度別に監査し、summary、case表、sensor表、熱収支、manifestは用途が異なるため維持した。
  COMSOL、monitor、notebook、外部report builderは基本経路に含めていない。

### Step 4 — 不要物を削除してv1境界を固定する（P1、完了）

- `rg`による参照、再生成手順、品質試験、正本の所在を確認してから削除する。
- 旧仕様書、旧presentation、外部report builder、派生Markdown reportの削除を確定する。
- 標準analysisに置換された個別plot、使われないconfig key、到達不能な互換処理を監査する。
- cohesiveなparser、validator、runnerはcomplexity値だけを理由に分割しない。
- 公開APIは `domain / io / engine / learning / inference / analysis / workflows` の責務を維持する。

受入条件:

- READMEから辿れない生成物・入口がない。
- quality gate、quickstart、TopCell、COMSOLの公開済みscreening結果が維持される。
- 削除した内容を別の抽象frameworkとして作り直さない。

実装結果:

- 旧仕様書と旧拡張guideは、正本である本計画、`product_architecture.md`、`extending.md`へ統合済みで、
  現行文書から参照されないことを確認して削除した。
- Git管理されていた5件のpresentation、外部HTML report builder、その複製source data、派生Markdown
  reportを削除した。今後のrepository直下の派生出力は`/outputs/`としてGit管理外にした。
- 旧`inference.py`は同名packageの`forecast / initialization / monitor / settings`へ責務移行済みで、公開
  re-exportを維持した。旧engine/inference試験も責務別試験へ移行済みのため重複fileを削除した。
- 現存するGit管理Markdownのlocal link、CLI入口、module参照を監査し、欠落linkと削除物への参照が
  ないことを確認した。
- quickstart、TopCell、COMSOLの3 configを監査した。現行keyは処理で使用され、未知key拒否も各sectionに
  あるため、根拠のないconfig互換性変更は行わなかった。
- 標準analysis図、sensor別表、熱収支、manifestは判断粒度が異なる正本なので維持した。file数だけを
  理由にparser、runner、benchmark outputを共通frameworkへ再統合していない。
- Ruff、Pyrefly、import-linter、Radon、coverageを含むPR gateと197試験が通過した。直前の同一worktreeの
  evidenceもTopCell 15/15、線形COMSOL 11/11を維持し、非線形・高忠実度結果は資格状態を明記して残した。

### Step 5 — CAE時間刻みと実験で資格化する（P1、実装完了・追加CAE/実測dataは後日）

開始に必要なのは、同一caseの基準刻み・細分刻みCAE、時刻・入力・sensorを対応付けた実測data、dataを
見る前に決めた用途別acceptanceである。これらがない間は空templateや追加modelで代用しない。

- high-fidelity過渡caseを少なくとも基準刻みと細分刻みで比較する。
- mesh差、time-step差、測定不確かさ、model errorを別々に保持する。
- `validate_experiment.py`へ実測値を投入し、補間せず同一key・controlで比較する。
- 用途別acceptanceはdataを見る前に利用者が決める。

受入条件:

- `mesh_qualified`、`temporal_qualified`、`experiment_compared`、`acceptance_passed`を混同しない。
- 全条件を満たすまで結果はscreening用途のままとする。
- model fitが良いことをCAE/実機資格の代用にしない。

現時点の実装:

- `run_time_step_convergence.py`を追加し、同一HV02入力・出力時刻で最大BDF刻み2 s / 1 sを比較する責務を
  外部COMSOL層へ限定した。通常dataset生成とcore RC計算の既定動作は変更していない。
- 受入値はsolve前に、領域平均・outlet 0.05 K、最高温度0.10 K、圧力差・放射熱量0.5%へ固定した。
- 合格した実データが揃った場合だけ`temporal_qualified`を更新し、時間刻み差を
  `temporal_uncertainty_*`としてmesh差・実験標準不確かさと分離して保持する。
- Java runnerは任意の最大BDF刻みを受け取れるようにし、公式COMSOL 6.4 compilerでcompileを確認した。
- 2 s / 1 sのraw solveは完了した。領域平均3項目と放射熱量の4/8項目が事前基準を超えたため、基準を
  緩めず`temporal_qualified=false`を維持する。0.5 sへの追加細分化は後日実施する。
- 実験templateは空欄であり、`experiment_compared=false`、`acceptance_passed`未設定のままとする。
- quickstart、TopCell、線形・非線形・high-fidelity COMSOLの用途、指標、資格状態、正本fileは
  [`benchmark_evidence.md`](benchmark_evidence.md)へ一つの読み口として整理した。
- 追加CAEを保留してもsoftware基盤v1、既存benchmark、通常の学習・予測・monitorは利用できる。保留中は
  high-fidelity結果をscreeningから設計保証へ昇格させず、Step 6で比較したmodelをproductへ採用しない。

### Step 6 — modelを増やすか判断する（P2、screening完了・product不採用）

Step 1–5の残差に、複数の独立caseで再現する構造がある場合だけ候補を追加する。

評価候補の順序:

1. operating-point別RC
2. 物理RCへ小さなresidual補正
3. Thermal Neural NetworkまたはSINDy
4. 大量・長履歴dataがある場合だけsequence model

採用条件:

- 同じsplit、同じcausal horizon、同じbaseline、同じ指標で比較する。
- meanだけでなくworst case、peak、外挿、安定性、再現可能な推論時間でRCを上回る。
- 改善しないmodel、option、専用前処理は削除する。
- observerは予測modelではないため、この順位表へ混ぜない。

実装結果:

- 過去に存在したMLP、1D-CNN、TCN、GRU、LSTMを`benchmarks/neural_comparison`へ評価専用で再実装した。
  productのconfig、artifact schema、CLI、公開APIは増やしていない。`TPU`というmodel実装は履歴にもないため、
  名称を捏造せずTCNを評価対象とした。
- 現行と同じcase split、外部case、可変`dt`、将来truthを使わないcausal open-loop境界で比較した。内部は
  train / validation / held-out testを明示し、全splitを時刻0の温度だけから予測した。外部は別directoryで、
  splitter・係数学習・epoch選択へ未投入であることをcase ID付きCSVへ固定した。
- 内部testのcase平均RMSEは、TopCellでRC 0.019 Kに対しニューラル群0.113–0.551 K、線形COMSOLで
  RC 0.027 Kに対し0.072–0.169 K、非線形COMSOLでRC 0.405 Kに対し1.291–2.767 Kだった。
- 外部coreのcase平均RMSEはTopCell、線形COMSOL、非線形COMSOL、high-fidelity COMSOLの全てでRCが最小だった。
  放射model-gapではニューラル5モデルの平均が改善したが、通常caseを大きく悪化させ、1 seedのscreeningでもある。
- clean学習済みmodelを固定し、内部testと外部coreの観測prefixへ0.15 K / 0.50 K noiseを各5 realization
  与えた。0.15 KでもRCは全7境界で最小だったが、これは推論時の観測noise感度だけで、未知物理への妥当性ではない。
- 物理RCのmodel-gap / core RMSE比は、TopCell温度依存熱損失68.7倍、非線形COMSOL放射23.9倍、
  high-fidelity放射3.0倍だった。通常caseの低誤差やfitted coefficientを物性同定・適用外保証へ使わない判断を明記した。
- 現時点ではニューラルmodelをproductへ採用せず、比較コードとweightをbenchmark責務内に留める。再検討は
  複数seedとphysics residualが同じ外部境界でRCを上回る場合だけ行う。

## 5. 現在のmodelの使い分け

| model / 推定器 | 役割 | 現時点の扱い |
|---|---|---|
| persistence | 最後の観測温度を維持する最低baseline | model採否の比較対象であり配備modelではない |
| engineering-prior RC | 未学習の設計初期値、sanity check | 学習改善量を測るbaseline |
| fitted physical RC | 解釈可能な同定とopen-loop予測 | 現在の主model。試験済み範囲のscreeningに使用 |
| Kalman observer | 観測同化、欠測、未知熱、sensor bias | monitorが必要な設備だけで使用 |
| MLP / 1D-CNN / TCN / GRU / LSTM | benchmark評価済み、product未採用 | 外部coreでRCを上回らず、比較責務内に限定 |

既存evidenceでは、TopCell、線形COMSOL、非放射の非線形COMSOL、高忠実度holdoutでfitted RCがpriorと
persistenceを上回る。一方、global-8放射caseは平均RMSE 4.556 K、worst 9.412 Kで、放射pair差も十分に
再現しない。比較境界と正本は[`benchmark_evidence.md`](benchmark_evidence.md)に集約した。したがって
fitted RCは一般的な「高精度model」ではなく、適用範囲を明示した熱時系列screening modelである。

## 6. 物理・データの不変条件

簡略化しても次は失わない。

- `temperature[k]`は`time[k]`、`commands[k]`は`[time[k], time[k+1])`に対応する。
- resampleを前提にせず、各区間の可変`dt`を使用する。
- edgeは一つのconductanceを共有し、内部熱流の相反性とenergy conservationを保つ。
- capacity、conductance、source係数、actuator時定数の正値制約を保つ。
- sensor数とnode数を分け、sensor mappingを観測行列として扱う。
- 学習、forecast、monitorは同じ状態方程式とartifactを使う。
- train/validation/testは行ではなくcaseまたはrecipe単位で分割する。
- truthは評価時だけ使用し、forecast requestへ混入させない。
- model scoreとmesh/time/experiment資格を別判定する。

## 7. 計画変更の規則

計画を変更するときは、同じ変更単位で次を更新する。

1. この文書の目的、優先順位、完了条件
2. READMEの利用順序と最初に読む成果物
3. config・CSV・artifact・公開APIへの影響
4. benchmark比較条件と既存結果への影響
5. 移行後に不要になるcode、test、文書、生成物の削除

変更理由が「complexity値を下げる」「将来使うかもしれない」だけなら実施しない。各実装単位は、利用者が
新しく得られる判断、または削除できる明確な負債のどちらかを持たなければならない。

## 8. 品質gate

各Stepで最低限次を実行する。

```powershell
uv run --locked python quality.py architecture
uv run --locked python quality.py pr
```

- Ruff: formatと局所lint
- Pyrefly: product境界の型整合
- import-linter: module依存方向
- Radon: 新しいD–F complexityとMI Cを導入しない
- pytest: unit/property/integrationとbranch coverage 88%以上

数値処理の変更では、同一入力で変更前後のCSV、JSON、主要指標を比較する。図だけの変更でも元表の値を
再計算しないことを確認する。COMSOL solveを伴わない変更では既存rawを再利用し、再solveを評価へ混ぜない。

## 9. 文献から採用する範囲

- 低次の物理modelは高速な同定・予測に使い、詳細CAEは空間場と外部妥当化に使う。
- wafer・chamberの評価では平均温度だけでなく、peak、温度均一性、昇温・冷却応答を見る。
- transient thermal impedanceは既知の吸収熱stepと定常性が確認できる場合だけRthへ解釈する。
- grey-box、TNN、ANNの優劣は対象dataと予測条件で変わるため、論文値ではなく同じ保持caseで採否する。

主な根拠:

- Schaper et al., *Dynamics and Control of a Rapid Thermal Multiprocessor*, 1992.
- Kim et al., *Sparse identification modeling and predictive control of wafer temperature in an atomic layer etching reactor*, 2024.
- Lee et al., *Plasma Ion Bombardment Induced Heat Flux on the Wafer Surface in ICP-RIE*, 2023.
- Székely and Van Bien, *Fine structure of heat flow path in semiconductor devices*, 1988.
- Kirchgässner et al., *Thermal Neural Networks*, 2021.

これらは設計原則であり、本repositoryの精度保証ではない。精度と用途はStep 1–5のevidenceだけで判断する。
