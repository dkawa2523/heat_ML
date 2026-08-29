# Thermal RC model notes

## 同定される量

- edge conductance: edgeごとの倍率
- actuator tau: actuatorごとの倍率
- source gain: sourceごとの倍率
- boundary conductance: boundaryごとの倍率

すべてengineering priorからのlog倍率として保持され、常に非負です。`learnable: false`を
設定した項はprior値に固定されます。

## 固定する量

- heat capacity
- edge topology
- source/boundary node weights
- source threshold
- boundary temperature intercept/slope
- sensor-node mapping

これらも同時に自由化すると、データだけでは互いを分離できない場合が多いためです。変更する
場合は、独立した熱容量測定、熱流測定、または十分に励起された実験設計を用意してください。

## 時間離散化

CSVの指令行は既定でleft-continuous zero-order holdです。actuatorは区間内で解析的に変化し、
温度forcingは中点近似、温度状態は区間内affine系の厳密解を使います。

TopCell synthetic generatorも区間内actuator midpointを使いますが、temperatureはforward Euler、
modelはexact integrationなので完全に同じ離散化ではありません。最新の外部評価値は
[TopCell benchmark](../benchmarks/topcell/README.md)に集約しています。実データでは、command
timestampが「開始時刻」か「終了時刻」かを必ず確認し、必要な場合だけ
`control_convention: right`を指定してください。

## 初期状態

sensorとnodeが一対一なら、初期node温度は観測値です。隠れnodeがある場合は観測写像のridge
逆問題を解き、未観測nodeを観測平均へ弱く寄せます。長いburn-inログがある場合はmonitorを
先に流し、そのfiltered stateをforecast初期値として使う拡張が適切です。

actuator初期値は既定で最初のcommandとし、開始前に定常だったと仮定します。開始直前の実効値
が既知なら、ライブラリAPIの`initial_actuator`へ渡せます。

## Loss

Huber lossはKelvinで計算します。sensor別標準化をしないため、静かなsensorだけが過大な重みを
持ちません。`huber_delta`は外れ値を二乗誤差から線形誤差へ切り替える温度幅です。

`prior_weight`は係数をengineering priorへ弱く戻します。これは物理式をlossへ重複実装する
penaltyではなく、不十分な励起に対する識別性regularizationです。

学習中のlog倍率は数値安定性のため`±4`に制限します。したがってengineering priorに対する
探索倍率はおよそ`e^-4`から`e^4`です。データごとの調整項ではなく、すべての学習に共通する
発散防止の境界なので、設定項目にはしていません。

## 適用外の兆候

次が全caseで同じ符号・温度依存性を持つ場合、微調整よりmodel termの追加を検討します。

- 高温域だけ残差が急増する: radiationまたは温度依存物性
- 加熱／冷却の履歴で同一指令への応答が異なる: hysteresis、phase change
- command停止後も遅い熱源が残る: actuatorを一階から多段stateへ
- 空間モードがsensorごとに一貫して残る: node/topology不足

単一caseのノイズに合わせて自由度を追加しないでください。運転条件単位のholdoutで再現する
残差だけを構造不足の根拠とします。
