# 熱時系列解析基盤の完成計画

この文書は目的、達成状況、次の実装判断を記録します。現行のAPI・出力・責務の正本は
[product architecture](product_architecture.md)と[extending](extending.md)、実行手順は
[README](../README.md)、数値と用途資格は[benchmark evidence](benchmark_evidence.md)です。
過去の実装順序は第5節へ履歴として残し、当時の出力配置や試験件数を現行契約として使いません。

## 1. 完成させるもの

CAEまたは実験の温度時系列と、時間変化する入熱・冷却条件から、同じ熱状態方程式で次を行います。

1. peak、昇温・冷却速度、温度均一性を解析し、明示したstep区間だけ応答時間・整定、適格なZth/Rthを求める。
2. engineering priorの熱容量・構造を使って熱経路、熱源gain、境界熱伝達、actuator遅れを同定し、
   保持caseのcausal誤差を示す。
3. 未知recipeから温度をopen-loop予測し、状態不確かさと適用範囲を保存する。
4. 必要な設備だけ、同じ熱回路へ観測を逐次同化し、未知発熱・sensor bias・欠測をmonitorする。

利用順序は`analyze -> train -> forecast`、monitorは任意です。扱うのは部品へ吸収された熱量、
または測定・校正・外部解析で対応が分かるcommandと冷却条件です。plasma/RFの電磁場、反応、吸収率、
CFDを本体で計算せず、装置固有条件は`system.yaml`とdataで表します。

## 2. 完成の定義と現在地

software基盤の完成と対象装置での妥当化を分けます。自己完結trajectory CSV、`system.yaml`、
`config.yaml`から解析・同定・予測を再現できる基盤は実装済みです。実機の精度保証には、用途別の独立dataと
事前基準が必要であり、screeningの低誤差や空templateで代用しません。

2026-10-05時点で、保存先・評価照合・任意診断・重複計算・依存関係に加え、入力設定と出力の簡素化、
専門解析configの分離、保存済みCOMSOLからの熱回路下書き・trajectory生成を完了しました。
最小config、熱回路不要の波形解析、衝突しないCSV列、共通読込設定、温度単位の引継ぎを公開経路へ反映しました。
全体品質gate、標準quickstart、TopCell 15/15判定、既存線形COMSOL 11/11判定の実行記録と効率化の計測値は
[quality](quality.md)へ集約し、試験件数を計画文書へ重複管理しません。次の実装判断は第4節に従います。

| 領域 | 達成状況 | 次に必要なもの |
|---|---|---|
| domain / IO / engine | 可変dt、欠測、hidden node、安定積分と数値境界を実装済み | 用途別dataで時間・memoryを確認 |
| 入力 / config | 最小例、型・重複列検査、共通CSV規約とunitを実装済み | 実験の入力単位・sensor対応を確認 |
| COMSOL導入 | 外部adapterで代表領域の容量・温度・入力を既存YAML/CSVへ変換 | 装置の領域縮約・熱結合・観測対応を確認 |
| analyze | 列名だけで基本波形・均一性、明示stepで応答、適格Zthを別出力 | 用途に必要なstep区間だけ指定 |
| train | case-balanced同定、causal選択、score・来歴・artifactの原子的保存を実装済み | 独立recipeで誤差と同定性を確認 |
| forecast | 因果的な観測prefixから予測、区間・coverage・指標を保存 | 必要horizonと許容誤差の確定 |
| diagnostics | 要求時の比較・熱経路・熱収支・図を基本出力と分離 | 用途上必要な場合だけ生成 |
| 出力 / 外部後処理 | sensor/node/controlのquantity列と単位を統一、外部評価は保存CSVを採点 | 旧case CSVを再生成、独自consumerの列名を更新 |
| monitor | 同じRCの観測同化・未知熱・bias推定を実装済み | 実機dataに基づくnoise・gaugeの確認 |
| TopCell / COMSOL / ニューラル比較 | 評価用screeningを実装済み | 数値・CAE資格の現在値はevidence参照 |
| 対象装置の妥当化 | 未完了 | 用途別mesh/time-step資格、実測比較、事前acceptance |

対象装置での完了条件は、対応するcase・時刻・入力・観測量でCAEと実験を比較し、sensor配置・時定数・
校正不確かさを記録することです。独立recipeでmean/worst誤差、peak、peak時刻、uniformity、
必要horizon・外挿条件を事前基準へ照合します。本モデル単独による安全保護、製品合否、hotspot保証は
範囲に含めません。

## 3. 今回の計画変更と派生影響

目的は維持し、基本の数値結果を確実に保存してから任意のdiagnosticsを生成する構成へ変更しました。
`project.diagnostics`はYAML booleanで既定`false`です。trainのscore・CSV・metadata・artifact、
forecastのcase CSV・summary・coverage・指標・manifestは常時保存します。比較、熱経路・mode、
test予測詳細、熱収支、図は`true`の場合だけ各出力の`diagnostics/`へ別transactionで保存し、
診断失敗でも基本出力を保持します。forecast診断は保存済みCSVをcase単位で読み、再推論しません。

analysisの既定出力先は学習runと兄弟の`<run_name>_analysis`です。配列analysisはTorchを必要とせず、
model inspectionだけ要求時にlazy importします。観測prefixの契約はdomainへ集約し、既存inferenceの
re-exportを維持しました。TopCellは保存済みworkflow出力を採点し、CAEの純粋比較と資格証拠の照合は
entry pointから一方向に利用します。

今回の追加改良では、共通CSV規約を`data`へ集約し、各処理の明示値だけを優先します。
forecast/monitorの読込とmanifestに同じ解決値を使い、別のparserや設定schemaは追加しません。
既存sampleはtrainとforecastの刻みが異なるため、forecastの`dt: null`を明示して可変刻みを維持します。
学習回数、split、観測data、受入基準は変えません。

基本波形にはpeak・速度・開始/終了温度を残し、任意recipeへstep応答・整定を自動付与する処理を廃止しました。
`analysis.sensors/controls`で熱回路なしの解析を可能にし、学習初期化の先頭観測条件は解析へ要求しません。
明示stepの要求区間と有効観測区間は別列へ保存します。温度unitはprojectからartifact・CSV付属metadata・図へ伝え、
数値を自動換算しません。

公開CSVは`sensor.<name>.<quantity>`等の末尾quantityへ統一し、熱収支もsource/boundary/node/totalを分離します。
edgeはartifactの定義順indexで識別し、名前の合成による衝突を避けます。manifest schemaは3です。
旧forecast/monitor CSVと独自consumerは更新が必要ですが、旧artifactの再学習は要求しません。
TopCell・COMSOL評価、診断CSV・図、テスト、利用文書を同じ変更単位で更新しています。
外部評価は入口で既存manifestの入力・artifact・解決済条件を照合し、処理途中へ診断gateを増設しません。

engineは検証済み計算の再利用、learningはfit内の入力Tensor・profile候補軌道の再利用、forecastは
共通physical rolloutによる平均とcovariance伝播の分離で重複計算を減らします。単位、commandの時刻対応、
保存則、causal境界、autograd、有限性の確認は維持します。汎用cacheや新しいmodel familyは追加しません。

COMSOL導入は本体の設定を増やさず、既存system形式のtemplateと領域・式の対応から2つの通常入力を生成する
外部adapterへまとめました。領域容量と蓄熱に対応した代表温度だけを自動抽出し、熱結合や実効係数は
templateへ残します。指令の境界値は元モデルのまま保持し、既存left/right規約で読みます。
README・責務文書・拡張場所を同時に更新し、再solveと読込、詳細CAE資格と取り込み成功を分けています。

## 4. 次の実装順序

1. **装置の導入準備を短くする（完了）。** 実行条件は`config.yaml`、全case共通の熱回路は`system.yaml`、
   時系列条件はCSVとする。専門解析は独立した`analysis.yaml`へ置く。設定継承やscenario登録は増やさない。
   熱回路で利用者が確認する中心は代表領域・熱結合・入熱/冷却・観測対応であり、solverやmeshの設定は混ぜない。
2. **COMSOLから熱回路の下書きを作る（完了）。** 外部adapterが、利用者のtemplateと代表領域の対応から
   熱容量・容量加重温度・既知入力を抽出し、既存形式の`system.yaml`とCSVを生成する。
   生成物は通常形式として編集・利用でき、保存済みモデルから通常解析までの接続を確認済み。
   領域の縮約、接続、実効conductanceや学習対象は確認が必要で、温度exportだけから一意に決めない。
   COMSOL再solveや本体へのCOMSOL依存は通常の読込に要求しない。
3. **用途別の装置dataで確認する。** 吸収熱・冷却・sensor・時間軸を対応付け、許容誤差、horizon、
   更新triggerを事前に決める。既存screeningから実機保証へ自動で昇格させない。
4. **実際の計算負荷を測る。** case数・ログ長・欠測・可変dt、利用するdevice/dtypeを使い、
   入力準備、同定、予測、保存、任意診断の時間とmemoryを分けて確認する。
5. **必要な制約だけを解く。** 一括読込が制約になる場合だけstreaming IOを実装する。
   同定性が不足する場合は入力励起・観測配置・独立校正を先に改善し、自由parameterを増やす前に
   case固有初期状態と共有係数を分離できるか確認する。
6. **再現する物理不足だけを拡張する。** 複数の独立problem/holdoutで同じ残差構造が再現した場合だけ、
   law・状態・integratorの変更を選ぶ。新modelは同じsplit・因果境界・baseline・複数seedで評価し、
   改善と運用上の必要性が示されるまでbenchmark内に留める。

file行数やcomplexity値だけを理由にparser、validator、runnerを細分化しません。parameter uncertainty、
sparse演算、階層同定、追加modelも、用途の制約とdataの根拠が揃った場合だけ再検討します。

## 5. 過去の実装計画と履歴

次は当時のStep 1–6の達成記録です。図・比較表を通常runへ常時生成する当初計画は、第3節の任意diagnosticsへ
変更済みです。過去の数値、case数、試験件数をこの文書へ複製せず、現在値はevidenceと品質gateで確認します。

| 旧Step | 実装したもの | 現行での扱い |
|---|---|---|
| 1: 学習結果を判断可能にする | 3 baseline、case/sensor誤差、causal/conditional区別、test波形・parity | 基本scoreと任意diagnosticsへ分離 |
| 2: 予測の主要図を完成させる | 温度区間・command・span図、熱収支CSV・図 | 保存済みforecast CSVから要求時に生成 |
| 3: 最小利用経路を確定する | quickstartのanalyze→train→forecastと主要成果物案内 | 現行手順・配置はREADMEを正本とする |
| 4: v1境界を固定する | 旧処理・派生reportの整理、公開7層、責務別試験 | 境界を維持し、frameworkとして作り直さない |
| 5: CAE・実験で資格化する | mesh/time-step比較、raw証拠照合、実験比較入口 | 実dataと用途別基準による資格は未完了 |
| 6: model追加を判断する | MLP / CNN / TCN / GRU / LSTMの公平なscreening | 現在はproduct不採用、benchmark責務内 |

現在の主modelはfitted physical RCです。engineering priorとpersistenceは採否のbaseline、
Kalman observerは必要な設備での運用機能です。通常caseでの低誤差を、放射などの適用外条件や
局所最高温度の保証に読み替えません。

## 6. 変更時に維持する条件

- `temperature[k]`は`time[k]`、`commands[k]`は`[time[k], time[k+1])`へ対応し、可変dtをそのまま使う。
- edgeの相反性、energy conservation、正値の物理係数、sensorとnodeの分離を維持する。
- train、forecast、monitorは同じ状態方程式とartifactを使い、将来truthを予測入力へ混ぜない。
- splitは行でなくtrajectory単位とし、recipe groupingの範囲は現行architectureへ従う。
- model scoreとmesh/time-step/experiment資格を別判定する。
- 基本出力と任意diagnosticsの失敗境界、入力・artifactの来歴を維持する。

計画変更と同じ変更単位で、API・config・出力配置、README、責務文書、benchmark境界、不要になる処理を
点検します。数値処理は同じ入力で予測・指標・gradientを比較し、図は保存済み正本の値を使います。
必要な品質gateと実行方法は[quality](quality.md)へ集約し、COMSOL solveを伴わない変更では既存rawを再利用します。

## 7. 設計原則の根拠

低次の物理modelは高速な同定・予測に、詳細CAEは空間場と外部検証に使います。平均温度だけでなく
peak、uniformity、昇温・冷却応答を確認し、ZthをRthへ解釈するのは既知吸収熱stepと定常性が確認できる場合だけです。
modelの採否は文献の順位でなく、同じ保持caseで決めます。

参考: Schaper et al. (1992)、Kim et al. (2024)、Lee et al. (2023)、Székely and Van Bien (1988)、
Kirchgässner et al. (2021)。これらは設計原則であり、本repositoryの精度保証ではありません。
