# Nonlinear Electronic Chip Cooling dataset quality report

## 判定

2026-08-30にCOMSOL 6.4.0.429で全27ケースを再計算しました。データ構造、時刻、観測mask、
物理診断、来歴は整合しています。本データはRCモデルの同定・open-loop予測・causal monitoring・
放射省略誤差の外部CAE評価に使用できます。一方、代表ケースでメッシュ収束していないため、製品の
絶対温度、hotspot限界、圧力損失を保証する設計truthには使用しません。

## 対象とgrain

- source: COMSOL Application Library `Electronic Chip Cooling`
- physics: 3D laminar conjugate heat transfer、温度依存air、任意のsurface-to-surface radiation
- public fidelity: physics-controlled mesh level 8
- grain: 1行 = 1 caseの1 output time
- primary key: `(case_id, time)`
- inputs: `chip_power` 0--16 W、`coolant_temperature` 10--40 degC、
  `inlet_air_velocity` 0.05--0.30 m/s

| role | cases | rows | 公開観測 |
|---|---:|---:|---|
| train | 10 | 910 | 全時刻 |
| forecast | 9 | 1,329 | 初期行のみ |
| monitor | 3 | 273 | 全時刻、固定seed 0.15 K noise |
| model_gap | 5 | 455 | 初期行のみ |
| **total** | **27** | **2,967** | |

通常26ケースは10秒刻み91行、短パルス1ケースは1秒刻み601行です。

## QA結果

| check | result |
|---|---:|
| 重複 `(case_id, time)` | 0 |
| 時刻逆行 | 0 |
| numeric truth欠損 | 0 |
| `maximum < volume average` の矛盾 | 0 |
| 負のinlet-to-outlet圧力差 | 0 |
| radiation modeと放射熱量の不一致 | 0 |
| 温度範囲 | 15.00--142.99 degC |
| 圧力差範囲 | 0.00589--0.07627 Pa |
| 熱収支残差 absolute 99%点 | 0.0240 W |
| 熱収支残差 absolute最大 | 0.1096 W |

最大熱収支残差は1秒刻み15 W pulseの切替直後で、最大入力の0.73%です。monitor観測noiseの全ケース
実測値は平均0.0073 K、標準偏差0.1528 Kで、設定0.15 Kと整合しました。未指令発熱はNM02だけ、
指令と実効流速の差はNM03だけに存在し、公開command自体には非公開異常を混入していません。

## 旧global mesh確認と現在の扱い

代表NT01の900秒時点は次のとおりです。COMSOLのmesh size levelは数値が小さいほど細かい設定です。

| mesh level | chip average | pressure drop |
|---:|---:|---:|
| 9 | 70.121 degC | 0.018515 Pa |
| 8 | 71.631 degC | 0.016749 Pa |
| 7 | 74.142 degC | 0.014537 Pa |

level 9→8のchip差は+1.510 K、level 8→7は+2.511 Kです。level 8→7の圧力差は13.2%低下し、
収束傾向を確認できません。このglobal level比較を再生成する旧後処理は退役させ、現在の判定経路は
fin間流路、wall boundary layer、chip/contact近傍を明示した`nonlinear_high_fidelity/`の局所mesh
収束評価へ一本化しています。本表はmedium-fidelityデータをscreening用途に限定した根拠として残します。

## 放射model-gap

同じcommandを与えた放射あり--なしペアのchip平均温度差です。

| pair | 900秒差 | peak absolute radiation |
|---|---:|---:|
| NR01 / NT01 power levels | -5.016 K | 1.879 W |
| NR02 / NT05 power ramp | -4.649 K | 1.319 W |
| NR03 / NT06 combined levels | -4.050 K | 1.309 W |
| NR04 / NF04 hot-low-flow | -12.453 K | 3.062 W |
| NR05 / NF09 hot-start | -15.148 K | 4.436 W |

高温・低流量端ほど放射省略差が大きくなりました。hot-startペアは各physicsで定常初期場を作るため、
初期場の差もmodel-gapに含みます。時系列の詳細値は `radiation_pairs.csv` を参照してください。

## 利用境界

利用可能なのは、モデル構造選択、同定・予測・監視workflowの比較、運転域内外の誤差傾向、放射あり／
なしの相対評価です。設計保証へ進む場合は、局所メッシュ収束、実装基板・接触抵抗・乱流／自然対流の
妥当化、および実験データとの同一評価境界での比較が残ります。
