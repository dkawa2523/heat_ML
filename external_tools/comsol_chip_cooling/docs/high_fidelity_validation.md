# 局所メッシュ・実験同一境界データセット

## 目的と位置付け

このデータセットは、`data/nonlinear/` のworkflow評価用CAEを設計判断へ近づけるための外部検証層です。
全領域を一律に細かくせず、温度・圧力損失を支配するfin間流路、壁面境界層、chip、thermal-grease
contact、heat-sink固体を個別に解像します。COMSOL依存は`external_tools/`内に閉じ、`celltemp`本体の
単純なRC基盤は変更しません。

この層が扱うのは次の3点です。

1. 隣接する局所メッシュ間で、熱・流動・放射の評価量が収束するか。
2. 採用メッシュのCAEを、本体と同じ`case_id, time, sensor, control`境界へ変換できるか。
3. 同じ境界の実測値が投入されたとき、補間せず不確かさ込みで比較できるか。

実測値が未提供の状態では3を「準備完了」と「妥当化完了」に分けます。空のtemplateやCAE値を実測値と
みなすことはありません。

## 局所メッシュ設計

元のApplication Libraryモデルは、physics-controlled meshと全壁共通の2層boundary layerを持ちます。
局所meshではfree-tetrahedralの下に選択別Sizeを置き、heat-sink壁と外側channel壁のboundary layerを
分離します。

| 制御対象 | coarse | medium | fine | 狙い |
|---|---:|---:|---:|---|
| air `hmax` | 10 mm | 7.5 mm | 6 mm | 主流とfin間流路 |
| heat-sink solid `hmax` | 2.0 mm | 1.5 mm | 1.1 mm | baseからfinへの伝導 |
| chip `hmax` | 1.2 mm | 0.9 mm | 0.65 mm | 4 mm厚chip内の勾配 |
| heat-sink wall `hmax` | 1.8 mm | 1.2 mm | 0.85 mm | fin面熱流束・流路幅方向 |
| chip/contact `hmax` | 0.8 mm | 0.6 mm | 0.4 mm | chipと薄層contact境界の面内温度・熱流束分布 |
| channel wall `hmax` | 8 mm | 6 mm | 5 mm | 外壁近傍流れ |
| sink boundary layer | 5層/1.5 mm | 6層/1.5 mm | 8層/1.6 mm | fin/base近壁の速度・温度勾配 |
| channel boundary layer | 4層/2.5 mm | 5層/2.5 mm | 6層/2.5 mm | channel no-slip壁 |

境界層の伸長率は1.2です。sink側第一層厚は順に約0.202、0.151、0.097 mmです。`mesh/*.csv`は
四面体・prism・pyramid数、skewness品質、局所サイズ、層数、総厚、第一層厚、生成時間を保存します。
point probeの値に合わせるのではなく、領域平均・境界平均・圧力差が収束することを採用基準にします。

Application Library原本のthermal greaseは50 umの物理厚みを持つ薄層contact境界（`sel2`、2次元選択）
として扱われ、独立した3次元領域ではありません。したがってcontactの`hmax`は界面の面内分割を制御し、
厚み方向へ四面体を積層する値ではありません。厚み方向の熱抵抗はcontact物理の厚みと熱伝導率で表します。

## 収束問題

過渡入力の違いを混ぜないため、次の2定常場を各メッシュで独立に解きます。

| case | 入力 | 覆う現象 |
|---|---|---|
| `MC01_nominal` | 12 W, 25 degC, 0.10 m/s, 放射なし | 定格熱抵抗、fin流路、圧力損失 |
| `MC02_hot_low_flow_radiation` | 12 W, 40 degC, 0.05 m/s, 放射あり | 高温低流量端、弱対流、表面間放射 |

隣接meshに対し、chip/base/fin平均温度は0.25 degC、chip/fin最大温度は0.50 degC、出口平均温度は
0.20 degC、圧力損失と放射熱量は相対2%をscreening許容値とします。これは製品の合否規格ではなく、
この外部評価データを生成するための離散化誤差基準です。結果は`mesh_convergence.csv`、差分は
`mesh_convergence_deltas.csv`、採否は`mesh_acceptance.csv`へ分離して保存します。

本コードのmodel-form errorを比較する用途には別のbenchmark基準を置き、領域平均温度0.50 degC、
最高温度0.75 degCまでを許容し、出口温度・圧力損失・放射熱量は上記と同じ基準にします。
`mesh_qualified`は厳格基準、`benchmark_qualified`はこの用途限定基準です。後者がtrueでもchip設計保証や
hotspot安全判定へ昇格させません。

`cae_reference.csv`は最細の実行済みmeshを同一実験境界へ変換した表です。直前meshとの差を
`mesh_uncertainty_*`に保存し、全ケースが各基準を満たした場合だけ`mesh_qualified`または
`benchmark_qualified`がtrueになります。

`dynamic/cae_reference.csv`は放射を含む`HV02`を実験比較用の過渡境界へ変換した表です。過渡条件は
MC01/MC02の発熱上限・入口温度・流速範囲内に置き、各温度のmesh不確かさには両定常点の
local-medium対local-fine差の最大値を保守的に付与します。ただし独立した時間刻み収束試験は未実施なので、
`temporal_qualified=false`を保持し、動的な設計保証には使いません。

## 2026-08-30の収束結果

| mesh | elements | min/mean skewness quality | MC01 solver | MC02 solver |
|---|---:|---:|---:|---:|
| global-8 | 48,870 | 0.0918 / 0.6376 | 30 s | 62 s |
| local-coarse | 837,644 | 0.0475 / 0.6577 | 223 s | 537 s |
| local-medium | 1,748,906 | 0.0652 / 0.6660 | 492 s | 1,380 s |
| local-fine | 3,587,721 | 0.0251 / 0.6770 | 1,169 s | 3,950 s |

global-8からlocal-coarseへ変えると、MC01のchip平均は15.913 degC、圧力損失は26.32%変わり、元meshを
絶対温度・圧力損失の設計値に使えないことが確認されました。local-mediumからfineへの差は次のとおりです。

| case | chip平均 | base平均 | fins平均 | 圧力損失 | 放射熱量 |
|---|---:|---:|---:|---:|---:|
| MC01 | 0.4962 degC | 0.4954 degC | 0.4922 degC | 0.624% | — |
| MC02 | 0.1544 degC | 0.1538 degC | 0.1491 degC | 0.905% | 0.130% |

したがってlocal-mediumは厳格0.25 degC基準には未達ですが、用途限定0.50 degC基準を両点で満たします。
本コードのmodel-form error評価用過渡データにはlocal-mediumを採用し、local-fineは日常生成に使わず
mesh不確かさを与える参照とします。`cae_reference.csv`はfine値を保持するため、
`mesh_qualified=false`、`benchmark_qualified=true`です。fineは平均品質が最も高い一方で局所最小品質が
0.0251へ低下しており、製品設計meshとして完成したとは判定しません。

## 非線形過渡pair

mesh評価と本体のmodel-form error評価を混ぜないため、過渡データは情報を集約した2ケースだけにします。
いずれも0–220秒を20秒間隔で出力し、時刻・3入力は完全に同一です。

| 区間 | chip power | inlet velocity | inlet temperature | 主に評価する応答 |
|---|---|---|---|---|
| 0–40 s | 0 W | 0.10 m/s | 25 degC | 定常初期場とゼロ入力整合 |
| 40–100 s | 0→12 W | 0.10 m/s | 25 degC | 発熱に対する熱容量・内部伝導 |
| 100–160 s | 12 W | 0.10→0.05 m/s | 25 degC | 流速依存対流と圧力損失 |
| 140–200 s | 12 W | 低流速側 | 25→35 degC | 高温・低流量側の結合非線形性 |
| 200–220 s | 12 W | 0.05 m/s | 35 degC | 複合端点での継続応答 |

`HV01_composite_conjugate`は共役熱流動、`HV02_composite_radiation`は同じ問題へ表面間放射だけを
追加します。前者は本体RCに対する非線形flow評価、両者の差はradiation model-gapです。入力は各20秒
区間でleft zero-order holdとして適用されるため、行`k`のcommandは`[t[k], t[k+1])`を駆動します。
これは元の27ケースを置換する学習全集合ではなく、高計算量mesh上で物理差を確認する外部評価pairです。

## 2026-08-31の過渡結果

| case | final/peak chip [degC] | max pressure drop [Pa] | max |radiation| [W] | max |heat residual| [W] | stationary/transient/class [s] |
|---|---:|---:|---:|---:|---:|
| HV01 conjugate | 45.0133 / 45.1386 | 0.010880 | 0 | 0.110454 | 424 / 4,136 / 4,638 |
| HV02 + radiation | 44.6320 / 44.7587 | 0.010883 | 0.395745 | 0.017481 | 629 / 4,832 / 5,754 |

220秒時点の`HV02 - HV01`はchip平均−0.3813 degC、base平均−0.3832 degC、fins平均
−0.3928 degCです。全時刻・3センサの最大絶対差も0.3928 degCで、出口空気温度は最大
+0.5210 degC、放射熱量の最大絶対値は0.3957 Wでした。

この温度差は保守的な隣接mesh不確かさ最大0.4962 degCより小さい値です。同一meshのpair比較では
共通離散化誤差が相殺され得ますが、過渡pair差そのもののmesh収束は確認していません。したがって放射が
heat sink温度を下げて空気側へ熱を移す方向性の確認には使えますが、0.39 degCを設計精度で確定した値とは
扱いません。

## 実験と同一の評価境界

一次キーと粒度は`(case_id, time)`です。比較列は`chip`, `sink_base`, `fins`、入力列は
`chip_power`, `coolant_temperature`, `inlet_air_velocity`です。CAEと実験でキーまたは入力が異なる場合、
`validate_experiment.py`は補間せず停止します。

CAEの3温度は領域平均なので、単一点thermocoupleとは同じ観測量ではありません。実験側は複数センサを
用いて同じ領域平均を近似し、その空間集約誤差を`uncertainty_*`へ含めます。具体的な受入列、配置原則、
反復条件は`data/nonlinear_high_fidelity/experiment/README.md`に定義します。比較時には実験標準不確かさと
mesh差を二乗和で合成し、bias、MAE、RMSE、最大誤差、正規化RMSE、95%不確かさ内率を出力します。
比較完了は受入合格と同義ではありません。`validation_status.json`は比較後も
`acceptance_passed: null`とし、用途別に事前設定した誤差・不確かさ基準を別途満たした場合だけ
`true`にします。benchmark報告も`experiment_compared`と`experiment_validated`を分けて表示します。

`experiment/experiment_template.csv`は過渡HV02と同じ時刻・3入力を持ち、温度・不確かさ欄だけを空欄に
した測定受入表です。`steady_experiment_template.csv`はMC01/MC02用です。空欄は欠測であり、0 degCや
CAE値による代入ではありません。

## 再生成

repository rootから実行します。

```powershell
uv run --with-editable . python `
  external_tools/comsol_chip_cooling/run_mesh_convergence.py --reuse-raw
```

単一meshの構造だけを確認する場合は次を使います。

```powershell
uv run --with-editable . python `
  external_tools/comsol_chip_cooling/run_nonlinear.py `
  --mesh-only --mesh-profile local-medium `
  --data-root external_tools/comsol_chip_cooling/data/nonlinear_high_fidelity
```

COMSOL Application Library原本は`loadCopy`で開き、上書きしません。schedule、raw table、solver logは
`work/nonlinear/mesh_*`へ置き、公開データと混在させません。
