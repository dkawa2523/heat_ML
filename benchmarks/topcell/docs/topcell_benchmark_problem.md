# TopCell benchmark definition

## 1. 本来の評価目的

対象は、少数温度sensorと運転commandから熱系を同定し、未知の運転履歴を安定に予測しながら、
実測時には状態・bias・innovationを因果的に更新する基盤です。したがって、学習済み軌道の
再生精度だけでは不十分です。

benchmarkは次を別々に評価します。

- identification: 定常gain、熱結合、境界、actuator tauを識別できる励振があるか。
- forecast generalization: 未知level、外挿、未知の時間波形、初期状態、可変`dt`。
- monitoring: noise基準、slow drift、欠測継続、sensor fault、未command熱負荷。
- adequacy diagnosis: 現在の線形RC構造で表現できない現象を検出できるか。

## 2. 学習データの役割

`work/data/raw/`だけが学習workflowから検出されます。

| 種類 | 数 | 目的 |
|---|---:|---|
| 4×4×4の定常運転点 × 3初期温度 | 192 | control gain、熱結合、境界、初期温度依存を同定 |
| 各actuatorの独立step | 18 | 応答遅れと符号を他入力から分離 |
| 各actuatorの独立ramp | 18 | stepだけに依存せず移動入力への追従を同定 |

複合recipe、短周期pulse、範囲外level、sensor faultは学習へ入れません。学習run内のrandom splitは
early selection用であり、最終的な汎化根拠には使用しません。同じcommand履歴で初期温度だけが
異なる軌道は`group_by_controls`により同じ内部splitへ入ります。

## 3. 外部forecastケース

各CSVの通常sensor列は初期観測だけを持ち、以後は空欄です。`truth_*`列は評価器だけが使用し、
modelやforecast workflowには渡りません。

| ID | group | 学習との差 | このケースで分かること | 一次指標 |
|---|---|---|---|---|
| F01 | interpolation | 全commandが格子中間値 | 単純な運転点内挿 | 全軌道RMSE、prior改善率 |
| F02 | interpolation | 加熱・冷却を同時に中間化 | source/boundaryの重ね合わせ | 全軌道RMSE、prior改善率 |
| F03 | extrapolation | brine低、heater/plasma高が範囲外 | 高入熱側の物理外挿と安定性 | RMSE、最大絶対誤差 |
| F04 | extrapolation | brine高、heater低が範囲外 | 強冷却側の外挿と符号 | RMSE、最大絶対誤差 |
| F05 | dynamics | 10秒幅の反復plasma pulse | plasma tauと誤差蓄積 | 全軌道RMSE |
| F06 | dynamics | 未知のphase順で3入力同時変更 | 複合recipe汎化 | 全軌道・終端RMSE |
| F07 | dynamics | 3入力の重なったsmooth ramp | 連続入力と複数tauの追従 | 全軌道RMSE |
| F08 | initialization | 初期温度70–85℃ | 学習初期pattern外からのcooldown | 全軌道RMSE |
| F09 | initialization | 初期sensorが4点中2点だけ | 初期状態復元不足の影響 | RMSE、観測sensor数 |
| F10 | sampling | 0.5–1.5秒の非均一刻み | resampleなし可変`dt`積分 | 全軌道RMSE、有限性 |
| F11 | model gap | 温度とともに増える熱損失 | 線形RCの適用限界を露出できるか | core群に対する誤差増幅 |

F11は低RMSEを要求しません。むしろmatched-physicsケースと同程度に見える場合、benchmarkの
非線形差が弱すぎるか、評価にリークがあると判断します。

## 4. 外部monitorケース

monitor CSVは実測相当sensor列、command、評価専用`truth_* / truth_bias_*`列を持ちます。

| ID | group | 注入する事象 | このケースで分かること | 一次指標 |
|---|---|---|---|---|
| M01 | baseline | sensor noise 0.15 Kのみ | 正常時innovationの基準幅 | residual RMSE |
| M02 | bias tracking | 初期0から緩やかなsensor drift | 物理温度とslow driftの分離 | filtered vs raw truth RMSE |
| M03 | fault detection | CP sensorへ+3 Kのstep offset | bias適応前の故障検出 | normalized residual、遅れ |
| M04 | missing data | Center/edgeの時間窓欠測 | 残sensorと熱結合による継続 | finite state、truth RMSE |
| M05 | disturbance detection | commandにないCP/Center熱負荷 | sensor故障でない物理外乱の検出 | normalized residual、遅れ |

絶対sensor offsetは、開始時に独立な基準温度がなければ`physical temperature + bias`から一意に
分離できません。そのためM02は初期biasを0とし、その後の変化を評価します。実機で絶対biasが
必要なら、校正点、冗長sensor、既知平衡条件のいずれかが必要です。

M05ではfiltered temperatureがtruthへ一致することを要求しません。モデルに存在しない熱源を
既知状態へ吸収するより、normalized innovationで異常を検出することが正しい目的です。

## 5. Truthとmodel mismatch

matched-physics truthは4 nodeの対称RC、plasma/heater source、brine/ambient boundary、一次actuator
lagを持ちます。

```text
thermal mass = [0.95, 1.15, 1.35, 1.55]
actuator tau = {brine: 4, heater: 8, plasma: 1} s
plasma gain = 0.010
heater gain = 0.014 above 70
brine boundary h = 0.030
ambient boundary h = 0.004
```

truth generatorは区間内actuator midpointとforward Euler、modelは解析actuator更新とexact thermal
integrationを使うため、完全に同一の離散モデルではありません。F11だけはambient conductanceを
温度依存にし、現在のmodel classでは表現できない構造差を意図的に加えます。

## 6. 自動合否条件

`evaluate_benchmark.py`は以下を判定します。

- F01–F10がすべて有限。
- F01–F10のcase平均RMSEが`1.0 K`未満。
- F01–F10の平均RMSEが未学習engineering priorを下回る。
- F11のRMSEが`max(0.5 K, core平均の2倍)`を上回り、model gapが可視化される。
- M01 residual RMSEが`0.35 K`未満。
- M02 filtered physical temperatureがraw measurementよりtruthへ近い。
- M03がnormalized residual 4以上で2秒以内に検出される。
- M04のfiltered stateが欠測中も有限。
- M05がnormalized residual 4以上で5秒以内に検出される。

これらはsynthetic sanity thresholdです。実機の許容温度、警報誤検知率、安全余裕を代替しません。
実運用thresholdは実データnoise、制御周期、熱許容差から再設定します。

parameter recoveryは副指標です。係数間に相関があっても外部軌道が正しく予測できる場合があり、
逆に係数が近くても未知波形で誤差が蓄積する場合があります。正値・有限性を必須とし、個別係数
誤差だけを合否にはしません。

## 7. 現時点で証明しないこと

- 実機、実材料、実CAE solverに対する精度。
- 放射、相変化、温度依存物性を含む非線形系の予測精度。
- 外部基準なしの絶対sensor offset分離。
- parameter uncertaintyを含む予測区間のcoverage。
- 隠れnode topologyの同定性能。engineの観測写像・hidden-state更新はunit testで確認しているが、
  本benchmarkは4 sensor / 4 nodeの係数・予測・監視問題に限定する。

必要になった機能をこの1ケースへ混在させず、別の明確なsystem benchmarkとして追加します。
