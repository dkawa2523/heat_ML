# Electronic Chip Cooling 非線形外部評価問題

## 1. 位置付け

このデータセットは、`celltemp` の少数ノードRCモデルを、学習に使った線形境界条件とは異なる
3D CAEへ投入して評価するための中忠実度ベンチマークです。COMSOLの結果を実機の設計保証値とは
みなしません。評価したいのは次の4点です。

1. 流速で変わる対流熱伝達と温度依存空気物性を、固定係数RCがどこまで近似できるか。
2. 未学習の入力履歴・運転端・短パルス・非一様初期場で、予測誤差がどう増えるか。
3. 未指令発熱と未指令の冷却能力低下を、通常の非線形応答や測定noiseと区別できるか。
4. 表面間放射を省略したモデルの系統差を、同一入力のCAEペアから定量化できるか。

線形データセットv1は `docs/problem_definition.md` に残し、本データとは混在させません。v1は
`storage + conduction/contact + linear convection` の同定確認、本データはmodel adequacyと外部妥当性の
確認を担当します。

## 2. CAE問題設定

出発点はCOMSOL 6.4 Application Libraryの
`Heat Transfer Module/Tutorials, Forced and Natural Convection/Electronic Chip Cooling`
（`chip_cooling.mph`）です。runnerは原本を `ModelUtil.loadCopy` で開き、原本を変更しません。

| 項目 | 設定 |
|---|---|
| 形状 | chip、50 um thermal-grease接触、aluminum heat sink、4 fins、矩形air channel |
| chip | 定数物性silicon: `k=148 W/(m K)`, `rho=2329 kg/m3`, `Cp=700 J/(kg K)` |
| heat sink | Application Libraryのaluminum |
| 流体 | Application Libraryの温度依存air material |
| 流れ | 3D Laminar Flow、fully developed inlet、pressure outlet、no-slip walls |
| 熱流動連成 | Heat Transfer in Solids and Fluids + Nonisothermal Flow |
| 入熱 | chip領域へ総発熱量を指定 |
| 放射比較 | Surface-to-Surface Radiationを追加した同一入力ペア |
| 放射率 | heat-sink walls `0.90`、channel walls `0.85` |
| 周囲放射温度 | channel開口を含め `coolant_temperature` と同じ時系列 |
| 観測温度 | chip、sink base、全finsの体積平均 |
| 補助truth | chip/fins最高温度、outlet平均温度、inlet-outlet平均圧力差、放射熱量、熱収支 |

入力は `chip_power [W]`、`coolant_temperature [degC]`、
`inlet_air_velocity [m/s]` の3つです。公開入力とは別に、監視ケースだけが
`hidden_power` または `effective_air_velocity` を持ちます。前者は未指令発熱、後者は指令値に現れない
冷却能力低下を表します。

point probeはmeshや座標変更に弱いため使わず、次の領域平均を観測にします。

| CSV列 | COMSOL selection | 検証体積 | 意味 |
|---|---|---:|---|
| `chip` | named `sel1`（final domain 4） | `6.4e-6 m3` | silicon chip平均温度 |
| `sink_base` | final domain 2 | `1.0e-5 m3` | heat-sink base平均温度 |
| `fins` | final domains 3, 5, 6, 7 | `1.9e-5 m3` | 4 fins全体の平均温度 |

runnerは毎ケースでこの体積を再計算し、Application Library更新などでselection対応が変わった場合は
公開前に停止します。

各ケースは、時刻0の入力を定数として定常共役熱流動を先に解き、その場を過渡解析の初期値として
引き継ぎます。通常ケースは一様温度に近い定常場、hot-startケースは発熱中の非一様定常場から開始します。
定常解析へ時間変数を持つ関数を渡さないため、初期化Studyと過渡Studyは分離しています。

過渡解析はBDFを使い、公開時刻をstrict stepとします。自動生成されたsegregated solverの収束許容値は
変えず、強連成の入力切替でも同じ基準へ到達できるよう時間依存解析の反復上限を25回とします。通常ケースは
900秒を10秒間隔、20秒パルスケースだけ600秒を1秒間隔で出力します。commandはrow `k`を区間
`[t[k], t[k+1])` に適用するleft zero-order holdです。

## 3. ケース構成

### 同定用: 10ケース

| ID | 主な励起 | 評価できること |
|---|---|---|
| NT01 | 0/4/8/12 W levels | 発熱level依存の熱抵抗・時定数 |
| NT02 | 12→3→10 W | 高温後の冷却と再加熱の非対称性 |
| NT03 | inlet 25→15→35 degC | inlet温度level依存 |
| NT04 | velocity 0.10→0.05→0.20 m/s | 流速依存の冷却conductance |
| NT05 | power ramp | step形状に依存しない連続応答 |
| NT06 | 3入力の非同期levels | 入熱・温度・流速効果の分離 |
| NT07 | 3入力の同時ramps | 運転域内部の連続被覆 |
| NT08 | 35 degC、0.05 m/s | hot/low-flow学習端 |
| NT09 | 15 degC、0.20 m/s | cold/high-flow学習端 |
| NT10 | 10 W定常場から開始 | 非一様hot-start |

入口温度と流速の単独rampは、それぞれNT03/NT04のlevelsとNT07のcombined rampに情報が重複するため
持ちません。連続入力への追従はNT05とNT07へ集約します。

### 外部forecast: 9ケース

| ID | 学習との差 | 評価できること |
|---|---|---|
| NF01 | 中間power・inlet温度 | 固定流速での内挿 |
| NF02 | 未知の3入力順序 | recipe汎化 |
| NF03 | 16 W | hot-side外挿 |
| NF04 | 40 degC、0.05 m/s、12 W | hot/low-flow複合外挿 |
| NF05 | inlet 10 degC | cold-side外挿 |
| NF06 | 0.075/0.15 m/s | 流速内挿 |
| NF07 | 0.30 m/s | 流速外挿 |
| NF08 | 20秒、15 W反復pulse | 欠落fast mode |
| NF09 | 未知の12 W定常場 | 非一様初期場からの予測 |

forecastの通常sensor列は初期行だけ値を持ち、以降は空です。全時刻の `truth_*` は評価専用であり、
open-loop予測の入力には使いません。

### Causal monitoring: 3ケース

| ID | 物理事象 | 評価できること |
|---|---|---|
| NM01 | 非線形通常運転 + 0.15 K noise | 正常時のfalse alert |
| NM02 | 420–520秒の未指令3 W発熱 | heat disturbance検出 |
| NM03 | 指令にない0.05 m/sへの一時流量低下 | cooling-loss検出 |

3ケースとも測定noiseは固定seedで再現できます。公開commandは正常値のままにし、評価専用列にだけ
実際の発熱・流速を保存します。

### Radiation model-gap: 5ケース

`NR01/02/03/04/05` はそれぞれ `NT01/NT05/NT06/NF04/NF09` と同じ入力です。違いは
表面間放射を解くことだけです。このペアにより、発熱level、連続sweep、3入力運転、hot/low-flow端、
非一様hot-startでの放射省略誤差を分離します。

## 4. メッシュと忠実度

公開データはCOMSOL physics-controlled mesh level 8を使います。代表ケースNT01では、level 8から
level 7へ細かくすると900秒時点のchip平均温度が2.51 degC上昇し、圧力差が13.2%低下しました。
level 9から8でもchip平均温度が1.51 degC上昇しており、細分化に対する変化は十分に小さくなっていません。
したがってmesh independenceは成立していません。

これは欠陥を隠すのではなく、用途境界として明示します。本データは以下に使えます。

- RC同定・forecast・monitoring workflowの外部CAE評価
- 入力域内外でのmodel-form error比較
- 放射有無の相対比較
- より高忠実度CAEまたは実験データへ進む前のscreening

次には使いません。

- chip/packageの設計認証や絶対温度保証
- 圧力損失の最終設計値
- hotspot安全限界の直接判定
- turbulence、自然対流、接触ばらつき、実装基板を含む製品予測

この課題に対して、wall boundary layer、fin間流路、chip/contact近傍を分離したcoarse/medium/fineの
局所mesh系列を`data/nonlinear_high_fidelity/`へ追加しました。2つの定常代表点で熱・流動・放射QoIを
隣接mesh間比較し、合格したprofileだけで情報集約型の過渡pairを生成します。詳細は
[`high_fidelity_validation.md`](high_fidelity_validation.md)を参照してください。元の27ケースは
screening用途の来歴を維持し、局所mesh結果で暗黙に置換しません。

## 5. 公開CSV

全CSVは単独で読め、manifestを必要としません。

| 分類 | 主な列 |
|---|---|
| 時刻 | `time` |
| 観測 | `chip`, `sink_base`, `fins` |
| 公開入力 | `chip_power`, `coolant_temperature`, `inlet_air_velocity` |
| 温度truth | `truth_chip`, `truth_sink_base`, `truth_fins`, `truth_chip_max`, `truth_fins_max` |
| 流体truth | `truth_outlet_air_temperature`, `truth_pressure_drop` |
| 物理診断 | `truth_radiative_heat_rate`, `truth_energy_residual` |
| 非公開事象 | `truth_hidden_power`, `truth_effective_air_velocity` |
| 来歴 | `case_id`, `case_group`, `fidelity`, `mesh_profile`, `mesh_size_level`, `source_model` |

training CSVは観測列が全時刻で埋まり、同値の平均温度truth列は重複させません。forecastとmodel-gapは
初期観測だけを公開し、評価用truthを持ちます。monitorはnoiseを含む観測とnoiseなしtruthを両方持ちます。

## 6. ディレクトリ

```text
data/nonlinear/
  train/                 10 identification trajectories
  eval/forecast/          9 open-loop trajectories
  eval/monitor/           3 causal-monitor trajectories
  eval/model_gap/         5 radiation trajectories
  qa_summary.csv          case-level QA evidence
  mesh_sensitivity.csv    representative mesh comparison
  radiation_pairs.csv     paired radiation effect summary
  quality_report.md       generated datasetの品質判定と用途境界
work/nonlinear/           git管理外のschedule、raw table、solver log、確認用mph
```

`qa_summary.csv` などは入力manifestではなく、生成結果を監査するための派生資料です。削除しても各CSVは
読め、runnerから再生成できます。

## 7. QA

生成時に次を自動確認します。

- 時刻の行数、一致、単調増加
- COMSOLへ実際に適用された5入力と公開scheduleのleft-ZOH一致
- named selectionと3領域体積の不変性
- 温度・入力・診断量のfinite値と工学的範囲
- roleごとの観測欠損規則
- case ID、fidelity、mesh、入力規約の来歴

熱収支残差、メッシュ差、放射ペア差は0に固定する契約ではなく、QA summaryで大きさを記録して用途判断に
使います。

## 8. 参照

- [COMSOL 6.4 Electronic Chip Cooling](https://doc.comsol.com/6.4/doc/com.comsol.help.models.heat.chip_cooling/chip_cooling.html)
- [COMSOL 6.4 Nonisothermal Flow coupling](https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_ug_multiphysics_features.12.12.html)
- [COMSOL Programming Reference Manual](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/COMSOL_ProgrammingReferenceManual.pdf)
