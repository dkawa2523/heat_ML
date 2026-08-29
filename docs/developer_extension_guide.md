# Developer extension guide

## 最も一般的な拡張: system.yaml

コード変更なしで次を追加できます。

- thermal nodeとheat capacity
- conductive edge
- actuatorとtau
- source、threshold、node分布
- boundaryとaffine boundary temperature
- sensor-to-node mapping

まずYAMLだけを変更し、`test` splitのworst-caseとセンサー別残差を比較してください。

## 新しいCSV形式

独自exportを直接engineへ読ませず、`celltemp.io`で`Trajectory`へ変換
します。変換後に守るのは次の4点だけです。

1. `time`は有限かつ単調増加。
2. 温度は`N`行、commandは`N-1`区間。
3. 欠測は`observation_mask=False`。
4. sensor/control名はsystem定義と同じ順序。

標準ローダーで扱える場合は、exportを改名せず設定済みdirectoryへ追加します。定数・時変を
問わずcommandはCSV列に置きます。独自のファイル名解析器や用途別Caseクラスは追加しません。

## 新しい物理項

温度依存項が必要な場合は、まず`ThermalRCModel.system_matrix`と`forcing`の責務を崩さない
独立した小さな項として実装します。次を試験してください。

- 単位と符号
- 係数制約
- 熱源なし受動系の安定性
- 可変`dt`での収束
- forward/backwardのfinite gradient
- holdout trajectoryでの改善

非線形温度項を入れると区間内affine厳密積分は使えません。implicit Newton、operator split、
substepのいずれを選ぶかを物理時間尺度に基づいて決めてください。

## 新しいobserver state

既存observerはthermal nodeとsensor biasを持ちます。初期時点から存在する絶対offsetは外部基準
なしにphysical temperatureと一意分離できないため、biasを絶対校正値とは扱いません。校正点が
ある場合や、装置別drift・遅い未観測熱源を追加する場合は、状態遷移、process noise、measurement
matrixを同じ拡張状態に組み込みます。推論後のEMAとして別処理を増やさないでください。

## 新しいworkflow

workflowはファイル読込・出力だけを担当し、数値処理を`engine`、`learning`、`inference`へ置きます。
公開CLIを増やす前に、既存の`train / forecast / monitor`の設定追加で表現できないか確認します。

## 完了条件

- unit: 対象物理量の解析解または不変量
- property: 広い正の係数／`dt`で破綻しない
- integration: artifact保存後の別loadで同じ結果
- evaluation: 運転条件単位のtestで平均とworst-caseが改善
- quality: format、lint、type、architecture、coverageが通る
