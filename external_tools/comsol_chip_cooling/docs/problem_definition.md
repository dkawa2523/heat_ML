# Electronic Chip Cooling 外部評価問題

## 1. 目的

この外部ツールの目的は、`celltemp` 本体と独立した実 CAE solver から、同じ温度・command 境界を
持つ時系列を作り、少数node RC基盤の同定・未知履歴予測・因果monitoringを評価することです。
COMSOLモデルを正解そのものと主張するものでも、実装をCOMSOL固有APIへ結合するものでもありません。

評価対象は次の問いです。

- 既知のchip発熱量とcoolant温度から、支配的な熱容量・熱結合・対流経路を同定できるか。
- 未学習のlevel、入力順序、短pulse、初期温度、可変samplingでopen-loop予測が破綻しないか。
- sensor noise、drift、offset、欠測、commandにない物理発熱を区別して観測できるか。
- 3D温度場を3nodeへ縮約したときの誤差を、学習内再現でなく外部trajectoryで露出できるか。

## 2. COMSOL問題設定

出発点は COMSOL 6.4 Application Library の
`Heat Transfer Module/Tutorials, Forced and Natural Convection/Electronic Chip Cooling`
（`chip_cooling.mph`）です。インストール済み原本は常に `ModelUtil.loadCopy` で開き、変更しません。

dataset v1は、RC構造と直接対応する最小の線形過渡問題に限定します。

| 項目 | 設定 |
|---|---|
| 形状 | tutorialのchip、thermal grease接触、aluminum heat sink、4 fins |
| chip | silicaを定数物性siliconへ変更: `k=148 W/(m K)`, `rho=2329 kg/m3`, `Cp=700 J/(kg K)` |
| heat sink | tutorialのaluminum: `k=238 W/(m K)`, `rho=2700 kg/m3`, `Cp=900 J/(kg K)` |
| 接触 | thermal grease `k=3 W/(m K)`, equivalent thin layer `50 um` |
| 外部放熱 | heat-sink外表面へ `h=10 W/(m2 K)` の対流熱流束 |
| 入力1 | `chip_power [W]`: chip volumeへの総発熱量 |
| 入力2 | `coolant_temperature [degC]`: 対流境界の外部温度 |
| 初期条件 | 指定した一様温度 |
| solver | Time Dependent, BDF、要求出力時刻をstrict stepとして使用 |

流体domain、Laminar Flow、Nonisothermal Flow、表面間放射は原本コピー内に残りますが、v1の
過渡studyではsolveしません。これは計算を軽くするためだけでなく、現行RCが表現する
`storage + conduction/contact + linear convection` を先に公平に評価するためです。流速依存対流、
放射、温度依存物性は、v1の合否と混ぜずmodel adequacy用の次段datasetにします。

## 3. 観測量と3node縮約

point probeはmeshや座標変更に脆いため使いません。全てvolume averageです。Application Library
モデルはair domain追加時に番号が変わるため、chipはCOMSOLの名前付き選択 `sel1` を直接使います。
最終geometryを体積・重心で確認した対応は次の通りです。

| CSV列 | COMSOL領域 | 物理的意味 | 固定熱容量 |
|---|---|---|---:|
| `chip` | named `Chip`（final domain 4） | silicon chipの体積平均温度 | 10.43392 J/K |
| `sink_base` | final domain 2 | heat-sink基部の体積平均温度 | 24.3 J/K |
| `fins` | final domains 3, 5, 6, 7 | 4 finsを合わせた体積平均温度 | 46.17 J/K |

`truth_chip_max`も診断列として出力しますが、RCのsensor/nodeにはしません。volume averageと最大温度の
差は、lumped modelを安全上の最大温度へそのまま読み替えられないことを示す縮約誤差です。

RC topologyは `chip -- sink_base -- fins` のchainです。発熱gainは既知の `1 W/W`、入力遅れは0、
熱容量はgeometry・物性から固定し、学習対象を2本の内部conductanceとbase/finsの2本の有効対流
conductanceだけに絞ります。既知量まで学習させて係数相関を増やす構成にはしません。

## 4. ケースが評価すること

### 同定用（16ケース）

| group | 数 | 分かること |
|---|---:|---|
| power steps | 5 | 入熱gainの確認、chip容量、内部時定数のlevel一貫性 |
| coolant steps | 4 | sourceと独立したbase/fins対流経路 |
| ramps | 2 | step形状だけへ過適合しない連続入力追従 |
| multilevel | 2 | 複数の時間scaleと運転levelでの一貫性 |
| combined | 2 | 2入力の重ね合わせと係数分離 |
| initialization | 1 | 単一初期温度への依存回避 |

### 外部forecast（8ケース）

| ID | 学習との差 | 評価内容 |
|---|---|---|
| F01 | 未学習の7 W | power内挿 |
| F02 | 2入力とも格子中間 | 重ね合わせ内挿 |
| F03 | 16 W（学習上限12 W超） | hot-side外挿と安定性 |
| F04 | 10 degC（学習下限15 degC未満） | cold-side外挿と符号 |
| F05 | 反復短pulse | 3nodeで欠落するfast mode |
| F06 | 未知の変更順・重なり | recipe汎化 |
| F07 | 一様50 degC開始 | 未知初期状態からのcooldown/reheat |
| F08 | 1/2/3秒の非均一時刻 | resampleなし可変`dt` |

forecast CSVの通常sensor列は初期行だけ値を持ち、以降は空です。全時刻の `truth_*` は評価専用で、
`celltemp forecast` には読まれません。

### 外部monitor（5ケース）

| ID | 注入事象 | 評価内容 |
|---|---|---|
| M01 | 0.15 K noiseのみ | 正常innovation幅 |
| M02 | chip/sinkのslow drift、finsは校正済み | referenceに対する絶対bias推定 |
| M03 | chip sensorへ+3 K step | NIS検出とsensor biasへの最終帰属 |
| M04 | base/finsの重なる欠測窓 | 残sensorと物理modelによる継続 |
| M05 | 3 Wの未command chip発熱 | NIS検出とunknown chip heatへの帰属 |

M01–M04は同じCOMSOL truthから測定事象だけを決定論的に作ります。M05だけはCOMSOL heat sourceへ
実際にhidden powerを加えて再solveします。`truth_bias_*` と `truth_hidden_power` は評価専用です。
`fins`を校正済みsensorとして`bias_reference`へ指定し、そのbiasを0へ固定します。M02/M03は
chip/sinkのoffsetをこのreferenceに対する絶対biasとして評価します。基準がない実データでは
零平均gaugeへ戻り、全sensor共通biasは一様な物理温度ずれと分離しません。

## 5. CSV契約

各CSVは `time, chip, sink_base, fins, chip_power, coolant_temperature` を単独で持ちます。COMSOL内の
commandはCOMSOLのpiecewise関数で左連続zero-order holdとし、公開CSVのrow `k`を区間
`[t[k], t[k+1])`へそのまま適用します。区間平均による近似変換を挟まず、本体の`left`規約と
CAE solverへ同一の入力波形を渡します。
最終rowのcommandは遷移に使われません。実行時に
別manifestを参照したり、file名からcontrolを復元したりしません。追加の `truth_*`, `case_id`,
`case_group`, `fidelity`, `source_model` は評価・来歴用であり、本体loaderは無視できます。

## 6. v1で証明しないこと

- 実chip/package/boardの材料、接触、風路に対する絶対精度。
- conjugate forced convection、流速command、放射、温度依存物性を含む非線形外挿。
- mesh independenceとsolver tolerance sensitivity。
- point sensorの設置位置・応答遅れ・校正誤差。
- CAEと実験間のmodel-form discrepancy。

次段では、同じCSV境界を保ったまま `(a) conjugate flow`, `(b) radiation`, `(c) 実験CSV` を別々の
外部評価群として追加します。v1へ混在させて何が原因の誤差か分からなくする構成にはしません。
