# 実験データ受入境界

実験値は存在しない値で補完せず、過渡用`experiment_template.csv`または定常用
`steady_experiment_template.csv`と同じ列で投入します。templateの時刻・入力は対応するCAEから生成し、
温度と不確かさだけを空欄にしています。主キーと粒度は
`(case_id, time)` の1行1評価時刻です。CAE側と実験側は、時刻と3入力が完全に一致する行だけを比較し、
補間や最近傍対応は行いません。

## 必須観測

| 列 | 単位 | 実験での定義 |
|---|---:|---|
| `chip` | degC | chip領域平均を近似する校正済み複数センサの重み付き平均 |
| `sink_base` | degC | heat-sink base領域平均を近似する複数センサの重み付き平均 |
| `fins` | degC | 4 fins全体の体積重みを近似する複数センサの重み付き平均 |
| `chip_power` | W | DUTへ実際に投入された電力。電源commandではなく電圧・電流から求める |
| `coolant_temperature` | degC | COMSOL inlet境界に相当する入口断面平均温度 |
| `inlet_air_velocity` | m/s | COMSOL fully-developed inletに相当する入口断面平均速度 |
| `uncertainty_*` | degC | 各集約温度の1標準不確かさ（校正、再現性、空間集約を含む） |

単一点のthermocouple値をCAEの体積平均へ直接対応させてはいけません。最低限、chipとbaseは中心・上流側・
下流側、finsは各finの根元・中間・先端を含む配置で空間勾配を確認し、集約重みと欠測処理を実験記録へ
固定します。圧力損失と出口温度を妥当化に使う場合も、CAEの入口・出口「断面平均」と同じになるよう、
圧力tap位置と温度traverseの集約方法を別途記録します。

## 実施条件

- channel、chip、heat sink、fin数・寸法を一致させ、50 um thermal greaseはCAEの薄層contactと同じ
  面積、厚み、熱伝導率で再現する。締結圧、塗布量、硬化状態も固定する。
- inlet温度と流量を定常化してから発熱を開始し、時刻原点を実投入電力の立上りへ合わせる。
- 入口はCAEのfully-developed断面平均速度に対応させる。断面速度分布、気圧、湿度を記録し、単一点の
  風速をそのまま`inlet_air_velocity`にしない。
- 温度・電力・流量を同じhardware clockで収録し、センサ時定数とlogger遅延を校正する。20秒評価時刻へ
  後処理補間せず、同期した実サンプルを残す。
- 各条件を独立に3回以上反復し、再現性を`uncertainty_*`へ含める。
- 放射ケースは表面仕上げと放射率測定値を記録し、未測定ならCAEの放射率を真値扱いしない。
- `run_id`、`sample_id` は追跡用であり、比較主キーには含めない。反復は事前に集約するか、別の
  `case_id` として扱う。

実データ取得後は次で同一境界を検査・比較します。

```powershell
uv run python external_tools/comsol_chip_cooling/validate_experiment.py `
  --cae external_tools/comsol_chip_cooling/data/nonlinear_high_fidelity/dynamic/cae_reference.csv `
  --experiment path/to/measured.csv `
  --output external_tools/comsol_chip_cooling/data/nonlinear_high_fidelity/experiment/result
```

現時点では一致する実測CSVが提供・公開されていないため、実験妥当化は未実施です。このディレクトリの
templateにある時刻・入力は試験条件ですが、空欄の温度・不確かさは実測データではありません。
