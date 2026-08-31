# Developer extension guide

## 最も一般的な拡張: system.yaml

コード変更なしで次を追加できます。

- thermal nodeとheat capacity
- constantまたは入力依存conductanceを持つinternal edge
- actuatorとtau
- scalar heat-rate lawとnode分布を持つsource
- reservoir温度、scalar conductance law、node分布を持つboundary
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

入力から非負の係数またはheat rateを作るだけなら、既存のscalar lawをedge、source、boundaryの
適切な位置へ配置します。特定装置部品や冷却方式の名前をlawへ追加しません。単一problemの残差に
しか効かない変換は外部評価側に残し、複数の独立problemで同じ数学的応答が必要な場合だけ共通lawを
追加します。
温度依存項が必要な場合だけ、`ThermalRCModel.system_matrix`と`forcing`で表せる区間affine経路と、
非線形heat-rate経路を分けます。学習・forecast・monitorで別モデルを作らず、次を試験してください。

- 単位と符号
- 係数制約
- 熱源なし受動系の安定性
- 可変`dt`での収束
- forward/backwardのfinite gradient
- holdout trajectoryでの改善

非線形温度項を入れると区間内affine厳密積分は使えません。implicit Newton、operator split、
substepのいずれを選ぶかを物理時間尺度に基づいて決めてください。

## 新しいobserver state

既存observerはthermal node、unknown heat、sensor biasを持ちます。基準なしでは零平均gauge、
校正済みsensorを`bias_reference`へ指定した場合はreference gaugeを使います。新しいbias表現を
並行実装せず、`bias_basis`、状態遷移、process noise、measurement matrixを同じ拡張状態として
変更してください。推論後のEMAとして別処理を増やさないでください。

## 新しいworkflow

workflowはファイル読込・出力だけを担当し、数値処理を`engine`、`learning`、`inference`へ置きます。
公開CLIを増やす前に、既存の`train / forecast / monitor`の設定追加で表現できないか確認します。

## 完了条件

- unit: 対象物理量の解析解または不変量
- property: 広い正の係数／`dt`で破綻しない
- integration: artifact保存後の別loadで同じ結果
- evaluation: 運転条件単位のtestで平均とworst-caseが改善
- quality: format、lint、type、architecture、coverageが通る
