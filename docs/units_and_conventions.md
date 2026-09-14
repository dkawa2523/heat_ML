# Units and conventions

設定、CSV、成果物で使用する単位と正規化規約をここへ集約します。装置固有の入力単位だけは
`system.yaml`側で定義し、モデル内部で暗黙変換しません。

| Quantity | Unit | Convention |
|---|---|---|
| time, `dt`, actuator `tau` | s | strictly increasing timestamps |
| temperature | °C or K | one consistent absolute scale per project; differences are K |
| heat capacity | J/K | positive and normally fixed during identification |
| edge/boundary conductance | W/K | non-negative scalar law output |
| source heat rate | W | non-negative commanded source output |
| unknown heat disturbance | W | signed heat in the observer disturbance basis |
| Huber delta, sensor/initial temperature std | K | temperature-error scale |
| disturbance process std | W/√s | continuous-time heat random-walk intensity |
| bias process std | K/√s | continuous-time sensor-bias random-walk intensity |
| sensor bias | K | relative to the reported bias gauge |

`sensor.node_weights`は非負かつ合計1で、node温度の点・面・体積平均を表します。一方、sourceと
boundaryの`node_weights`は熱量または熱伝達分布の係数であり、非負ですが合計1を要求しません。
したがって、同じ`node_weights`という名前でも正規化の意味は異なります。

monitorの未知発熱basisはsourceがあればそのsource weight行です。sourceが一つもない系では、
各nodeへ独立に発熱を置ける`node_identity`へフォールバックします。実際に使ったbasis名は
`run_manifest.json`へ保存されます。source分布が互いにほぼ線形従属する場合は外乱を一意に分離できない
ため、system定義側で物理的に区別できる最小の分布へ整理してください。

forecastの`temperature_std`と95%区間は、潜在的な物理温度の状態・process不確かさです。将来の
測定noise、同定パラメータ、入力、model-formの不確かさは含みません。
