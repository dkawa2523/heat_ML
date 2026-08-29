# Product architecture

## 1. 目標状態

この基盤は「CAE過渡履歴を近似するモデル集」ではなく、次の一つの熱状態を中心にします。

```text
CSV / DataFrame
      ↓
        Trajectory ──────────── ThermalSystemSpec
      ↓                              ↓
      └──────── ThermalRCModel ──┘
                 ↓          ↓
           full rollout   Kalman observer
                 ↓          ↓
              forecast    monitor
```

学習、予測、監視で別々の遅れ処理、rollout、補正器を持たないことが重要です。

## 2. Domain

### CSV discovery and split

学習入力は一つのdirectoryからpattern一致するCSVを自動検出します。各CSVがtime、sensor、
controlを持つため、別索引、ファイル名regex、暗黙の定数control抽出はありません。
ファイルを追加・改名しても同期作業は不要です。

分割は次の2方式だけです。

- `random`: time gridとcontrol履歴が同じtrajectoryを自動的に同じgroupへ入れる。
- `explicit`: 必要な評価だけ、任意の`case_id,split`表を使用する。

### Trajectory

- `time: [N]`
- `temperature: [N, n_sensor]`
- `commands: [N-1, n_control]`
- `observation_mask: [N, n_sensor]`

`commands[k]`は`[time[k], time[k+1])`にのみ対応します。可変時間刻みは
`dt = diff(time)`として保持し、再サンプリングで情報を落としません。
学習、forecast、monitorはいずれもこの同じ型を受け取り、schedule専用の並行表現は持ちません。

### ThermalSystemSpec

- thermal nodeと正のheat capacity
- 方向を持たないconductive edge
- commandと一次遅れtauを持つactuator
- threshold、gain、node分布を持つheat source
- affine温度とnode別熱伝達を持つboundary
- sensorからnodeへの観測写像

sensorはstate nodeの部分集合または別名です。これにより、3本のTCしかなくても4点以上の
内部状態を表現できます。

## 3. Engine

### 熱方程式

各edge `(i,j)`は単一の`G_ij > 0`を共有し、Laplacianを構成します。

```text
q_ij = G_ij (T_j - T_i)
C_i dT_i/dt = Σ_j q_ij + q_source,i + q_boundary,i
```

閉じた伝導系では`Σ C_i T_i`を保存します。境界と熱源を除いた受動系は、任意の正の
`dt`で初期温度の最小値・最大値を越えません。

source:

```text
q_source = gain · max(a - threshold, 0) · weights
```

boundary:

```text
T_boundary = intercept + slope · actuator
q_boundary = h · weights · (T_boundary - T)
```

### Actuator state

commandをそのまま熱源へ入れず、各actuatorを解析解で更新します。

```text
a_next = u + (a - u) exp(-dt/tau)
```

`tau=0`は遅れなしです。熱区間内のforcingにはactuatorの中点値を使用します。

### Integrator

温度について`dT/dt = A T + b`となるため、既定はaugmented matrix exponentialです。

```text
T_next = Phi(dt) T + Gamma(dt) b
```

`A`を逆行列化しないため、閉じた熱回路でも安定です。同一trajectory内の同一`dt`は
`Phi/Gamma`を再利用します。大規模系向けにはA-stable implicit Eulerも選べます。
単一軌道もbatch次元を1として`forward_batch`を通すため、学習用と推論用に別rolloutはありません。

## 4. Identification

正の物理係数は`prior * exp(log_multiplier)`で学習します。edgeごとに一つの係数しか
持たないため、学習で非対称熱流にはなりません。

学習単位は1ステップΔTではなくmultiple-shooting区間です。

1. caseを選ぶ。
2. `horizon`全体を確保できる観測時刻をshooting pointとして選ぶ。軌道自体が短い場合は、
   先頭観測から利用可能な最大区間を使う。
3. actuator stateを先頭からその時刻まで再生する。
4. 観測写像の逆問題からnode初期温度を得る。
5. `horizon`区間を自己回帰せず状態方程式で連続積分する。
6. 観測mask上のHuber lossをKelvin単位で計算する。

validationは完全な軌道で計算し、caseごとのRMSE平均でmodel stateを選択します。短いcaseや
観測点の多いcaseだけが過大な重みを持たない設計です。

capacityは既定で固定します。`C`とすべての`G/q`を同じ倍率で変える尺度不定性を避け、
同定されたconductanceとsource gainを解釈可能に保つためです。

## 5. Forecast

forecastは初期sensor観測をnode状態へ写像し、以降はcommand scheduleだけでopen-loop積分
します。安定性はclipではなく、正のcapacity/conductanceと安定積分で確保します。

入力はtrain、monitorと同じtrajectory CSVです。先頭行だけに初期温度を置き、以降の温度を
空欄にします。その1ファイルが初期状態、時間軸、将来commandをすべて表し、将来のsensor値が
混在した入力は拒否します。条件一覧からscheduleファイルを参照する二段構成は持ちません。

出力はsensor温度に加え、未観測node温度とeffective actuatorを持ちます。parameter uncertainty
を含まない状態分散だけを予測区間として見せることは避け、forecastは検証可能な物理軌道へ
限定します。

## 6. Monitor

monitorは後付け補正ではなく、拡張状態`[T, sensor_bias]`を持つKalman observerです。
入力は実測温度と適用commandを同じ行に持つtrajectory CSVであり、log一覧表は持ちません。

- `T`: RC方程式で予測
- `sensor_bias`: slow random walk
- measurement: `H T + bias`

出力は一ステップprior、filtered physical temperature、bias、innovation residual、予測標準
偏差です。欠測sensorはupdateから外し、他のsensorと熱結合を使って状態を継続します。

## 7. Artifact

artifactは次の3ファイルのみです。

- `model.pt`: `state_dict`。読込時は`weights_only=True`。
- `system.yaml`: topology、capacity、prior、観測写像。
- `metadata.json`: schema、integrator、同定後物理係数、データ範囲、評価結果。

前処理object、外部graph CSV、元configへの相対参照を持ちません。directory単独で移送できます。

## 8. Config and output boundary

用途ごとに1つの`config.yaml`だけを持ち、学習・forecast・monitorが共有します。すべての相対パスは
configの親ディレクトリ基準、乱数seedはtop-levelの1箇所です。出力は隣接する一時directoryへ
全ファイルを書き終えてから置換するため、入力不正や処理失敗で直前の正常出力を壊しません。

sensor/control名は`system.yaml`を唯一の定義元とし、configへ重複させません。artifactも既定では
`project.output_dir/project.run_name/artifact`から導出し、別runを読む場合だけ明示します。

## 9. Dependency direction

```text
cli
  → workflows
    → artifact / learning / inference
      → engine / io
        → domain
          → config
```

domainはnumpy以外のframeworkに依存しません。engine/learning/inferenceはpandas/YAMLを読みません。
CSV・YAMLと数値計算の境界を明確にしています。

## 10. Benchmark boundary

学習run内のrandom testだけを汎化性能とはみなしません。同一generator、同一level、同一recipeが
学習側にあれば、低RMSEでも未知運転への有用性を示せないためです。標準benchmarkは
`benchmarks/topcell/work/data/`内で学習directoryと外部評価directoryを物理的に分け、次を
個別に評価します。

- 未学習levelの内挿・外挿
- 学習にないpulse、複合recipe、smooth ramp
- 初期温度外挿、初期sensor欠測、可変`dt`
- noise、drift、sensor fault、欠測、未command熱負荷
- 現modelでは表現不能な温度依存熱損失のnegative control

monitorの`bias`は絶対校正値ではありません。開始時から存在する一定offsetは外部基準なしに
physical temperatureと一意分離できないため、benchmarkは初期基準後のdriftを評価します。
未モデル化熱負荷ではfiltered stateのtruth一致ではなくinnovationによる検出を評価します。
合成truthの数値定義は`scripts/definition.py`に集約し、generatorとevaluatorが共有します。

## 11. 今後追加する場合の優先順位

1. 複数seedまたはLaplace近似によるforecast parameter uncertainty。
2. 温度依存物性、放射、相変化を必要な系だけへ追加できるphysical term interface。
3. 長大trajectory／多数node向けのsparse matrix exponential。
4. 複数装置を扱う場合のhierarchical parameter sharing。

新しいNNモデルや別rolloutを横に増やすことは優先しません。現RC構造で系統的な残差が
説明できず、データ量と外挿評価が追加自由度を正当化した場合だけ検討します。
