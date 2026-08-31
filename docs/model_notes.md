# Thermal RC model notes

## 同定される量

- edge conductance scalar lawの係数
- actuator tau: actuatorごとの倍率
- source heat-rate scalar lawの係数
- boundary conductance scalar lawの係数

すべてengineering priorからのlog倍率として保持され、常に非負です。`learnable: false`を
設定した項はprior値に固定されます。

## 固定する量

- heat capacity
- edge topology
- source/boundary node weights
- positive-part lawのthreshold
- reservoir temperature intercept/slopeとcontrol
- power-lawのcontrolとreference
- sensor-node mapping

これらも同時に自由化すると、データだけでは互いを分離できない場合が多いためです。変更する
場合は、独立した熱容量測定、熱流測定、または十分に励起された実験設計を用意してください。

## 時間離散化

CSVの指令行は既定でleft-continuous zero-order holdです。actuatorは区間内で解析的に変化し、
`exact`では温度とactuatorを結合した区間内affine系の厳密解を使います。入力依存conductanceと
sourceのpower-lawを`exact`で使う場合、そのcontrolは区間内一定となる`tau=0`です。遅れを持つ
入力依存係数は`implicit`で区間中点の実効入力から組み立てます。この可否判断はsystem定義ではなく
数値engineの責務です。
一次遅れactuatorがpositive-part source thresholdを
横切る区間は解析的な交差時刻で分割するため、時間刻み固有の中点forcing近似はありません。

TopCell synthetic generatorは区間内actuator midpointとforward Euler、modelは結合系のexact
integrationなので完全に同じ離散化ではありません。最新の外部評価値は
[TopCell benchmark](../benchmarks/topcell/README.md)に集約しています。実データでは、command
timestampが「開始時刻」か「終了時刻」かを必ず確認し、必要な場合だけ
`control_convention: right`を指定してください。

## 初期状態

sensorとnodeが一対一なら、初期node温度は観測値です。時刻0だけを使う最小forecastでは、
隠れnodeを観測写像のridge逆問題で観測平均へ弱く寄せます。連続するburn-in観測をCSV先頭へ
置けば、forecastは物理observerで未観測nodeとeffective actuatorを因果推定し、履歴末端の
posterior物理状態からopen-loopへ引き渡します。将来区間の未知熱・sensor biasを既知と仮定
しないよう、observerの外乱・bias状態はhandoffしません。

actuator初期値は既定で最初のcommandとし、開始前に定常だったと仮定します。開始直前の実効値
が既知なら、ライブラリAPIの`initial_actuator`へ渡せます。

## Loss

Huber lossはKelvinで計算します。sensor別標準化をしないため、静かなsensorだけが過大な重みを
持ちません。`huber_delta`は外れ値を二乗誤差から線形誤差へ切り替える温度幅です。

既定の学習は全軌道rolloutです。各case内で観測点の平均を取り、さらにcase間で均等に平均するため、
長いログやsensor数の多いcaseだけが支配しません。`horizon`は長大ログの計算量を制限する明示的な
optionであり、通常の同定精度を調整するknobとしては使いません。

`prior_weight`は係数をengineering priorへ弱く戻します。これは物理式をlossへ重複実装する
penaltyではなく、不十分な励起に対する識別性regularizationです。

学習中のlog倍率は数値安定性のため`±4`に制限します。したがってengineering priorに対する
探索倍率はおよそ`e^-4`から`e^4`です。データごとの調整項ではなく、すべての学習に共通する
発散防止の境界なので、設定項目にはしていません。

## 適用外の兆候

次が全caseで同じ符号・温度依存性を持つ場合、微調整よりmodel termの追加を検討します。

- 高温域だけ残差が急増する: radiationまたは温度依存物性
- pressure、flow、recipe入力ごとに残差が変わる: 入力依存scalar law、または状態・空間分布の不足
- 加熱／冷却の履歴で同一指令への応答が異なる: hysteresis、phase change
- command停止後も遅い熱源が残る: actuatorを一階から多段stateへ
- 空間モードがsensorごとに一貫して残る: node/topology不足

単一caseのノイズに合わせて自由度を追加しないでください。運転条件単位のholdoutで再現する
残差だけを構造不足の根拠とします。
