# High-fidelity dataset quality report

## 判定

- local-medium strict mesh qualification: `False`
- local-medium benchmark qualification: `True`
- adjacent-mesh temperature uncertainty maximum: `0.4962 degC`
- experiment validation: `not performed (no matching measurements supplied)`
- transient time-discretization qualification: `not performed`

## 動的CAE

| case | rows | end [s] | T min/max [degC] | chip final/peak [degC] | max dp [Pa] | max |energy residual| [W] |
|---|---:|---:|---:|---:|---:|---:|
| HV01_composite_conjugate | 12 | 220 | 25.000/45.013 | 45.013/45.139 | 0.010880 | 0.110454 |
| HV02_composite_radiation | 12 | 220 | 24.998/44.632 | 44.632/44.759 | 0.010883 | 0.017481 |

## 放射pair

- maximum absolute sensor-temperature delta: `0.3928 degC`
- maximum absolute radiative heat rate: `0.3957 W`
- temperature delta / conservative adjacent-mesh uncertainty: `0.792`
- time and all three public inputs: identical
- quantitative resolution: `screening only; paired transient mesh refinement not performed`

## 用途境界

本データは本コードの非線形model-form error評価用です。strict mesh基準は未達のため、
chip設計保証、hotspot安全判定、圧力損失の最終設計値には使いません。実験templateの
時刻と入力はCAE境界から生成済みですが、温度・不確かさ欄は空であり実測値として数えません。
