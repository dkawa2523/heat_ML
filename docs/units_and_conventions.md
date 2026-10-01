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
| transient thermal impedance, effective thermal resistance | K/W | sensor temperature rise divided by an independently known absorbed-heat step |
| terminal Zth drift | K/(W s) | linear slope over the configured terminal window |
| Huber delta, sensor/initial temperature std | K | temperature-error scale |
| disturbance process std | W/√s | continuous-time heat random-walk intensity |
| bias process std | K/√s | continuous-time sensor-bias random-walk intensity |
| sensor bias | K | relative to the reported bias gauge |

熱収支ではsourceとboundaryの符号を「thermal nodeへ入る向きが正」とします。したがって冷却中の
`boundary_*_w`は負です。`edge_<node_a>_to_<node_b>_w`は`node_a`から`node_b`へ流れる向きを正、
`storage_<node>_w`は`C*dT/dt`を正とします。`balance_residual_*_w`は
`internal + source + boundary - storage`で、数値的に0であるべきです。

各actuatorは`system.yaml`で任意の`unit`と`role`を持てます。`unit`はCSVと図表に表示する入力単位、
`role`は人が入力目的を読むための短い説明です。どちらも自由文字列で、coreは単位変換や計算分岐に
使用しません。同じproject内では一つのcontrolを一つの単位に固定してください。推奨roleは
`heat_input`、`reservoir_temperature`、`heat_transfer`、`cooling_command`ですが、列挙制約では
ありません。省略した既存systemもそのまま読めます。

`sensor.node_weights`は非負かつ合計1で、node温度の点・面・体積平均を表します。一方、sourceと
boundaryの`node_weights`は熱量または熱伝達分布の係数であり、非負ですが合計1を要求しません。
したがって、同じ`node_weights`という名前でも正規化の意味は異なります。

monitorの未知発熱basisはsourceがあればそのsource weight行です。sourceが一つもない系では、
各nodeへ独立に発熱を置ける`node_identity`へフォールバックします。実際に使ったbasis名は
`run_manifest.json`へ保存されます。source分布が互いにほぼ線形従属する場合は外乱を一意に分離できない
ため、system定義側で物理的に区別できる最小の分布へ整理してください。

forecastの`temperature_std`と95%区間は、潜在的な物理温度の状態・process不確かさです。将来の
測定noise、同定パラメータ、入力、model-formの不確かさは含みません。

`thermal_paths.csv`はnode capacityをJ/K、edge/boundary conductanceをW/K、resistanceをK/W、
source heatをWで保存します。lawのoffset/scaleはそのpath出力単位を基準とし、threshold/referenceは
併記したcontrol unitを基準とします。`thermal_modes.csv`のpoleは1/s、time constantはsです。両表の
代表運転点は、成分別中央値に最も近い実在の学習command行を定常actuator値として評価したものです。

`thermal_impedance.csv`の`heat_step_w`は、生のcommand差ではなく対象へ吸収された熱量の変化です。
`analysis.thermal_impedance.steps`へcaseごとに明示し、coreは`unit`や`role`から換算しません。過渡値は
`Zth(t)=DeltaT(t)/heat_step_w`です。終端windowの正規化勾配が設定上限を超える場合、記録は定常熱抵抗を
確定できないため`effective_rth_k_per_w`を空欄にします。

予測比較のpeak時刻誤差は、真値peakが記録区間の内部にあるsensorだけで評価します。真値peakが先頭または
末尾なら、記録外のpeak時刻を特定できないため`peak_time_qualified=false`とし、case集約の
`max_abs_peak_time_error_s`から除外します。peak温度誤差は同じsensorでも引き続き報告します。
