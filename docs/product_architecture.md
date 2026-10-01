# Product architecture

## 1. 目標状態

この基盤は「CAE過渡履歴を近似するモデル集」ではなく、次の一つの熱状態を中心にします。

```text
CSV / DataFrame ── Trajectory ── ThermalSystemSpec
                         ↓              ↓
                         ThermalRCModel
                         ├─ full rollout ───────────── train
                         ├─ history observer ─ state ─ forecast rollout
                         └─ causal observer ────────── monitor
```

学習、予測、監視で別々の遅れ処理、rollout、補正器を持たないことが重要です。

利用者の基本経路は`analyze -> train -> forecast`です。`monitor`は逐次観測同化が必要な設備だけで使い、
基本経路の前提にしません。各workflowの出力は「dataset/case要約 → 判断用の主要図・比較表 → sensor別・
履歴・manifest等の詳細」の順で読みます。新しいindex生成層やreport builderは置かず、各workflowが
自分の出力directoryと正本CSV/JSONを所有します。

## 2. Domain

### CSV discovery and split

学習入力は一つのdirectoryからpattern一致するCSVを自動検出します。各CSVがtime、sensor、
controlを持つため、別索引、ファイル名regex、暗黙の定数control抽出はありません。
ファイルを追加・改名しても同期作業は不要です。

分割は次の2方式だけです。

- `random`: time gridとcontrol履歴が数値的に同じtrajectoryを自動的に同じgroupへ入れる。
- `explicit`: 必要な評価だけ、任意の`case_id,split`表を使用する。

`random`の同一判定は既定で相対・絶対とも`1e-9`の許容値を使い、CSV往復などの丸め差で反復caseが
分離することを防ぎます。より大きいlogger揺らぎを同一recipeとして扱う場合だけ
`split.recipe_rtol/recipe_atol`を単位と収録精度に合わせて明示します。

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
- scalar lawをconductanceとして持つ、方向のないinternal edge
- commandまたは計測入力と一次遅れtauを持つactuator
- scalar lawとnode分布を持つheat source
- reservoir温度則、conductance scalar law、node分布を持つboundary
- sensorから単一nodeまたはnode加重平均への観測写像

sensorはstate nodeの部分集合・別名、または合計1の非負node weightsです。これにより、3本のTCしか
なくても4点以上の内部状態を表現でき、CAEの面積・体積平均もnodeを増やさず対応できます。

`io/system.py`はsystem schema全体の読込・書出しを所有し、sensor項目の単一node・加重平均のYAML変換だけを
内部`io/_sensor_yaml.py`へ委譲します。加重和の有限性・非負性・合計1とnode参照整合性は引き続き
`ThermalSystemSpec`が検証し、YAML helperは公開APIや別のsensor domain型を増やしません。

scalar lawは熱機構名を持たず、入力から非負の1値を作る責務だけを持ちます。

actuatorの`unit`と`role`は任意の表示metadataで、単位変換や物理式の分岐には使いません。これにより
装置入力の意味を成果表へ渡しながら、装置名の列挙をcoreへ持ち込みません。

- `constant`: `value`
- `positive_part`: `gain * max(control - threshold, 0)`
- `power_law`: `offset + scale * (max(control, 0) / reference)^exponent`

同じ型をedge conductance [W/K]、source heat rate [W]、boundary conductance [W/K]へ配置します。
backside gas、接触、冷媒、RF、plasmaなどをcoreの列挙型へ追加せず、装置側のnode・path・入力で
構成します。

## 3. Engine

### 熱方程式

各edge `(i,j)`は単一の`G_ij(a) > 0`を共有し、Laplacianを構成します。

```text
q_ij = G_ij (T_j - T_i)
C_i dT_i/dt = Σ_j q_ij + q_source,i + q_boundary,i
```

閉じた伝導系では`Σ C_i T_i`を保存します。境界と熱源を除いた受動系は、任意の正の
`dt`で初期温度の最小値・最大値を越えません。

scalar lawを`lambda(a)`と書くと、sourceは次です。

```text
q_source = lambda_source(a) · weights
```

boundary:

```text
T_reservoir = intercept + slope · temperature_control
G_boundary = lambda_boundary(a)
q_boundary = G_boundary · weights · (T_reservoir - T)
```

内部edgeも`G_edge=lambda_edge(a)`として同じlawを使います。reservoir温度controlとconductance
controlは独立です。`exact`では入力依存conductanceとsourceのpower-law controlを`tau=0`に限定し、
時変係数を中点値で近似して厳密積分と呼ぶことを避けます。遅れを含む入力依存係数は`implicit`が
区間中点で評価します。positive-part sourceは一次遅れ入力でもactive区間ごとにaffineなため
exact積分できます。この解法適合性はdomainではなくengineが検証します。

`ThermalRCModel.heat_flow_breakdown`は同じ係数から、edgeの有向熱流、sourceとboundaryのnode別熱流、
`C*dT/dt`、node別収支残差をWで返します。熱流から求めたnet入力と状態空間RHSから求めた蓄熱率を
別経路で比較するため、表示用の自明な差し引きではなく実装整合の確認にもなります。

公開`ThermalRCModel`は係数と安定したAPIを所有し、`engine/operators.py`がLaplacian、forcing、結合
affine operator、`engine/rollout.py`がthreshold event、単一step、batch rolloutを所有します。
保存されるparameter/buffer名と公開methodはこの分割の影響を受けません。

### Actuator state

commandをそのまま熱源へ入れず、各actuatorを解析解で更新します。

```text
a_next = u + (a - u) exp(-dt/tau)
```

`tau=0`は遅れなしです。

### Integrator

sourceのactive/inactiveが固定された区間では、温度とactuatorを結合した
`x = [T, a]`が`dx/dt = F x + c`となるため、既定はaugmented matrix exponentialです。

```text
x_next = Phi(dt) x + Gamma(dt) c
```

`F`を逆行列化しないため、閉じた熱回路や遅れなしactuatorを含む特異系でも安定です。
一次遅れactuatorがsource thresholdを横切る場合は解析的な交差時刻で区間を分割します。
同じ`dt`、source activity、system matrixを変えるlaw入力を持つbatchは`Phi/Gamma`を再利用します。大規模系向けには
A-stable implicit Eulerも選べます。
単一軌道もbatch次元を1として`forward_batch`を通すため、学習用と推論用に別rolloutはありません。

## 4. Identification

正の物理係数はscalar law内部で`prior * exp(log_multiplier)`として学習します。edgeごとに一つのlawしか
持たないため、学習で非対称熱流にはなりません。

学習単位は1ステップΔTではなくcaseの軌道です。

1. 1 epochで全学習caseを一度ずつshuffleして処理する。
2. 既定では先頭観測から軌道末尾まで状態方程式を連続積分する。
3. shooting-point観測行列のnull空間だけを選択区間への線形感度からcase固有のnuisance stateとして
   profileする。直接・重複・加重平均sensorのいずれでも初期観測は保存し、弱観測modeには物理温度幅の
   Gaussian priorを置く。
4. 観測mask上のHuber lossをKelvin単位でcaseごとに計算する。
5. 軌道長や観測数では重み付けせず、case lossを均等に平均する。

非常に長いloggerデータだけ`horizon`を指定できます。その場合も各caseをepochごとに一度扱い、
caseごとに独立した有効長と観測可能なshooting pointを使います。短いcaseが同じbatch内の長いcaseを
切り詰めることはありません。

validationは設定した学習horizonに関係なく完全な軌道で計算し、先頭観測だけからの因果的な
open-loop RMSE平均でmodel stateを選択します。条件付きprofile RMSEも保存し、係数fitと初期状態推定を
監査できます。長いcaseや観測点の多いcaseだけが過大な重みを持たない設計です。

`workflows/train.py`は設定解決、trajectory読込、case split、同定の調停だけを所有します。同定後の
case/sensor評価、保持caseのfitted RC・engineering prior RC・persistence因果比較、validation overview、
学習範囲metadata、thermal path/mode表、CSV/JSONとartifact保存は
`workflows/train_output.py`が所有します。評価表を束ねる内部型は同moduleの外へ出さず、利用者向けAPIや
artifact schemaを増やしません。`learning`は表やfilesystemへ依存しません。

capacityは既定で固定します。`C`とすべての`G/q`を同じ倍率で変える尺度不定性を避け、
同定されたconductanceとsource heat rate係数を解釈可能に保つためです。

## 5. Analysis

`celltemp.analysis`はNumPy配列から波形・均一性・誤差・熱回路指標を計算する純粋関数だけを持ちます。
CSV/YAMLの読込、設定解決、表と図の保存は`workflows`側に残し、学習済みmodelへの依存を追加しません。
そのため、生のCAE/実験trajectoryとforecast結果に同じ指標定義を適用できます。

thermal impedanceは一般recipeから自動検出しません。利用者がcase、対象control、transition区間、既知の
吸収熱変化を明示した場合だけ、基準温度のdrift/noise、入力の単独性、sample数を確認して
`Zth(t)=DeltaT/P`を計算します。commandの表示用`unit`/`role`から吸収熱へ換算しないため、RF電源指令や
dimensionless操作量を誤ってWとして扱いません。定常Rthは終端Zth勾配がproject固有の明示閾値を満たす
sensorだけに出し、有限時間の最終値を無条件に定常値と呼びません。

## 6. Forecast

forecastは先頭から連続するsensor履歴を物理observerへ通し、履歴末端のposterior node温度と
effective actuatorから、command scheduleだけでopen-loop積分します。先頭1行だけの観測も
同じ処理の最小ケースです。安定性はclipではなく、正のcapacity/conductanceと安定積分で
確保します。

開始直前の実効actuatorが既知なら、trajectory CSV先頭行の`initial_effective_<control>`を使用します。
省略時だけ最初のcommandへ整定済みと仮定します。

入力はtrain、monitorと同じtrajectory CSVです。観測は先頭から連続するprefix、最初の
全sensor空欄行以後は将来です。個別sensorの欠測は許しますが、forecast境界後に観測が再登場する
入力は拒否します。その1ファイルが履歴、時間軸、将来commandをすべて表し、条件一覧から
scheduleファイルを参照する二段構成は持ちません。出力はposterior handoff時刻から始まり、
履歴の再構成値をopen-loop予測として数えません。

出力はsensor温度、全node温度、effective actuatorに加え、observerのposterior covarianceと
unknown-heat process noiseをopen-loop伝播した標準偏差・95%区間を持ちます。これは状態・process
uncertaintyだけで、parameter、将来入力、model-form uncertaintyを含みません。またartifactに保存した
学習control・sensor温度範囲と将来schedule・予測温度を比較し、case・quantity単位の適用範囲を
一つの別表へ保存します。この区間はGaussian observer仮定に基づき、経験的coverageを保証しません。
process noiseは保持データの残差を使って用途ごとに調整します。

同じ出力directoryへcase/sensor/control単位の標準波形指標を保存します。forecast workflowがanalysis
workflowを呼ぶのではなく、両者が`celltemp.analysis`の純粋関数を共有するため、workflow間の依存や
暗黙の追加設定はありません。

case別の主要予測図`figures/forecast_<case_id>.png`は、保存対象と同じforecast DataFrameから予測sensor
温度・95%状態区間・command・sensor spanだけを表示します。図のためにmodelを再実行したり、将来truth、
誤差、設定optionを追加したりしません。

熱収支は`energy_balance.csv`を数値の正本とし、同じ表からcase別の
`figures/energy_balance_<case_id>.png`を出力します。図はsource・boundaryの符号付き熱流、
`source + boundary`とstorage、数値残差の3段だけを示します。engineは熱流計算、workflow出力層は
表と可視化を所有し、図のためのmodel計算、設定option、workflow間依存は追加しません。

状態推定の公開importは`celltemp.inference`で維持し、内部は`settings`、`initialization`、`forecast`、
`monitor`へ分けます。forecastとmonitorは同じobserver設定解決と初期化関数を共有しますが、互いを
呼び出しません。

## 7. Monitor

monitorは後付け補正ではなく、拡張状態`[T, unknown_heat, sensor_bias]`を持つKalman observerです。
入力は実測温度と適用commandを同じ行に持つtrajectory CSVであり、log一覧表は持ちません。

- `T`: RC方程式で予測
- `unknown_heat`: `system.yaml`の既知source分布に沿うsigned heat rate [W]。sourceがない場合だけ
  node identity basisを使う
- `sensor_bias`: 明示したgauge内で識別可能なslow random walk
- measurement: `H T + sensor_bias`

未知熱とsensor biasの連続時間process noiseは、`exact`では熱方程式を含むVan Loan離散化、
`implicit`では同じbackward-Euler遷移を通す離散化で共分散へ変換します。
出力はprior physical temperature、predicted measurement、posterior physical temperature、
reconstructed measurement、node未知熱、sensor bias、bias gauge、innovation、full innovation
covariance、NISです。
欠測sensorはupdateから外し、残るsensorと熱結合を使って状態を継続します。

単一sensorのgross innovationはraw covarianceから計算するNISへそのまま残します。一方、状態更新では
標準化innovationが`innovation_gate_sigma`を越えた観測のnoiseを連続的に膨らませます。hard rejectや
事後clipではなくJoseph形式の共分散更新にも同じ有効noiseを使うため、異常検出を鈍らせず、一過性の
sensor faultが物理温度・未知熱へ直結する影響を抑えます。

全sensor共通offsetと一様な物理温度ずれは外部基準なしでは識別不能です。`bias_reference`を
省略した通常モードでは`sum(sensor_bias)=0`のgaugeを使います。校正済みsensor名を明示した場合だけ
そのsensorのbiasを0へ固定し、残るsensor offsetを基準に対する絶対値として推定します。出力列
`bias_gauge`は`zero_mean`または`reference:<sensor>`であり、数値の意味を暗黙にしません。
reference sensorがログ全体で一度も観測されない入力は、基準として機能しないため拒否します。

## 8. Artifact

artifactは次の3ファイルのみです。

- `model.pt`: `state_dict`。読込時は`weights_only=True`。
- `system.yaml`: topology、capacity、prior、観測写像。
- `metadata.json`: schema、integrator、同定後物理係数、データ範囲、評価結果。

読込時は`system.yaml`から導出した固定buffer、`model.pt`から再計算した同定後係数、両ファイルの
SHA-256をmetadataと照合し、同じshapeでも内容や来歴が異なる混成artifactを拒否します。

前処理object、外部graph CSV、元configへの相対参照を持ちません。directory単独で移送できます。

## 9. Config and output boundary

用途ごとに1つの`config.yaml`だけを持ち、学習・forecast・monitorが共有します。すべての相対パスは
configの親ディレクトリ基準、乱数seedはtop-levelの1箇所です。出力は隣接する一時directoryへ
全ファイルを書き終えてから置換するため、入力不正や処理失敗で直前の正常出力を壊しません。
相対`output_dir`はconfigの親directory配下に限定し、外部storageは絶対パスで明示します。
project root、そのancestor、filesystem rootそのものは置換対象にできません。既存結果はbackupへ
移してから新結果へ切り替えるため、置換失敗時にもrollbackできます。

sensor/control名は`system.yaml`を唯一の定義元とし、configへ重複させません。artifactも既定では
`project.output_dir/project.run_name/artifact`から導出し、別runを読む場合だけ明示します。
forecastとmonitorは同じobserver設定解決と初回観測更新を使い、入力・artifact hash、解決済み設定を
共通の`run_manifest.json`へ保存します。

## 10. Dependency direction

```text
cli
  → workflows
    → analysis / artifact / learning / inference
      → engine / io
        → domain / config
```

domainとconfigは互いに依存しないleafです。domainはnumpy以外のframeworkに依存しません。
analysis/engine/learning/inferenceはYAMLを読みません。engine/learning/inferenceはpandasも読まず、
CSV・YAML、指標計算、物理計算の境界を明確にしています。

## 11. Responsibility boundary

本体は装置物理のカタログではなく、熱時系列を解析し、観測可能な集中定数熱モデルを
同定・予測・監視する実行基盤です。
責務を次の4境界に固定します。

- `src/celltemp`: trajectory、熱network、入力応答、数値積分、同定、状態推定、artifact。
  装置部品名、CAE製品、個別case、合否閾値を知りません。
- `system.yaml`とproject data: 対象系のnode、heat path、入力、sensor写像を定義します。装置ごとの差は
  まずここで表現し、本体classを装置名で増やしません。
- `external_tools`: CAE・実験の実行、export変換、geometryやmesh、problem definition、データ品質を
  所有します。本体はCOMSOLや実験設備へ依存しません。
- `benchmarks`: 学習から分離した評価集合、指標、閾値、negative controlを所有します。benchmarkを
  通すためのcase別補正を本体へ戻しません。

線形COMSOL評価では、`evaluate.py`がscreening基準、summary、成果表保存、CLIを所有し、
`evaluate_forecast.py`がartifact再計算・prior/persistence比較、`evaluate_monitor.py`が
NIS・sensor bias・未知熱eventの評価を所有します。両者は表を返すだけで、互いを呼び出さず、
合否値やfilesystem出力も持ちません。

mesh収束評価では、`run_mesh_convergence.py`のCLIがcase/profileを選び、runtime準備、mesh QoI収集、
evidence公開を順に調停します。runtime準備だけが既存raw・meshの再利用可否とCOMSOL起動を判断し、
QoI収集は既存`run_nonlinear.py`のsolve/mesh機能を再利用します。収束閾値と隣接mesh比較は同じ評価module、
原子的なCSV保存は公開処理に閉じるため、solver実行と合否判定を混在させません。

過渡時間刻み収束では、`run_time_step_convergence.py`だけが同一HV02 scheduleを初期pairの最大BDF刻み
2 s / 1 s、必要時は次の隣接細分化で実行し、同一出力時刻の8 QoIを比較します。solver optionは既存
`solve_case`へ任意値として渡し、通常生成の既定動作を変えません。各pairの履歴と資格判定に使う最新pairを
分け、時間刻み差はmesh差・実験不確かさと別列で保持してcore modelへ持ち込みません。

非線形COMSOL生成では、`run_nonlinear.py`のCLIがcase/profileを選び、runtime準備後にmesh統計または
trajectory datasetのどちらか一方へ振り分けます。runtime準備だけがraw再利用とCOMSOL起動を判断し、
dataset公開処理がrole別CSV、QA、放射pair、全catalog更新時の不要CSV整理を所有します。
`solve_case`、`solve_stationary_case`、`raw_tables_available`、`inspect_mesh`、`build_dataset`はmesh収束・
高忠実度runnerからも使う安定した外部tool内境界として維持し、core packageへは公開しません。

高忠実度benchmarkでは、`benchmark_high_fidelity.py`が学習・forecast実行、prior/persistence比較、
mesh差との比較、放射pair、summary判定を所有します。`benchmark_high_fidelity_output.py`は完成した表・
予測・summaryを受け、CSV/JSON、残差図、directory単位の置換だけを所有します。出力moduleはmodelや
判定基準を知らず、評価moduleはplot backendや成果物配置を知りません。

広域14ケースの非線形benchmarkでも、`benchmark_nonlinear.py`がcase読込、3 baselineの予測・比較、
残差・放射pair指標、summary判定を所有し、`benchmark_nonlinear_output.py`が6表、14予測CSV、2図、
summaryのstaged公開だけを所有します。高忠実度側とはschema、図、mesh資格が異なるため、共通report
frameworkには統合しません。両出力moduleは評価値を再計算せず、外部report builderにも依存しません。

`domain`は物理構造と参照整合性、`engine`は解法適合性と時間発展、`learning`は係数同定、
`inference`は状態推定、`analysis`は波形指標、`workflows`は入出力のオーケストレーションだけを
担当します。

## 12. Core extension rule

本体拡張は、次の順で必要性を切り分けます。

1. 入力列、node、edge、source、boundaryの組合せで表せるなら、system定義だけを変更する。
2. CAE・実験exportの違いなら、外部adapterで`Trajectory`へ変換する。
3. 特定caseだけの差なら、評価結果として残し、本体へ補正項を追加しない。
4. 独立した複数のproblem definitionとholdoutで同じ数学的不足が再現した場合だけ、共通law、状態、
   integratorのどこを拡張するか決める。
5. 追加自由度について単位、保存則、安定性、識別可能性、artifact再現性を本体テストへ追加する。

したがって、放射、相変化、接触、流体、特定装置部品などの名称そのものは拡張理由になりません。
本体が扱うべき新しい数学的構造と、複数対象での外部評価根拠が揃った場合だけcore interfaceを
変更します。将来候補のparameter uncertainty、sparse演算、階層同定も同じ基準で判断します。
