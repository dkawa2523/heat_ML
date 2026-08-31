# Nonlinear Electronic Chip Cooling dataset

COMSOL 6.4 Electronic Chip Coolingモデルを、共役層流熱伝達の時間依存問題として再計算した
self-contained trajectory CSVです。問題設定とケースの意味は
[`../../docs/nonlinear_problem_definition.md`](../../docs/nonlinear_problem_definition.md)を参照してください。

```text
train/                  10 identification trajectories
eval/forecast/           9 open-loop trajectories
eval/monitor/            3 causal-monitor trajectories
eval/model_gap/          5 radiation trajectories
qa_summary.csv           1 row per trajectory
mesh_sensitivity.csv     NT01 at mesh levels 7, 8, and 9
radiation_pairs.csv      paired radiation-minus-base effects
quality_report.md        dataset-level QA findings and use boundary
```

各trajectoryは別manifestを必要としません。基本列は次のとおりです。

| 列 | 単位 | 内容 |
|---|---|---|
| `time` | s | 時刻 |
| `chip`, `sink_base`, `fins` | degC | 公開観測 |
| `chip_power` | W | 公開chip発熱command |
| `coolant_temperature` | degC | 公開inlet-air温度command |
| `inlet_air_velocity` | m/s | 公開平均inlet流速command |
| `truth_*` | 列名に対応 | 評価専用のCAE真値・診断量 |

trainingは全観測を持ちます。forecastとmodel-gapは初期行だけ観測を持ち、以降は空です。monitorは
0.15 Kの固定seed noiseを含む観測とnoiseなしtruthを持ちます。非公開発熱と非公開流量低下は
`truth_hidden_power`、`truth_effective_air_velocity` にのみ現れます。

通常ケースは10秒間隔で91行、短pulseケースは1秒間隔で601行です。commandはrow `k`を
`[t[k], t[k+1])`へ適用するleft zero-order holdです。公開データはmesh level 8で、mesh independenceは
成立していません。本データはモデル評価・screening用であり、製品の設計保証値には使いません。

局所meshによる収束評価と実験受入境界は、別用途として
[`../../docs/high_fidelity_validation.md`](../../docs/high_fidelity_validation.md)および
`../nonlinear_high_fidelity/`へ分離しています。元の27 trajectoryを高忠実度に見せかけて置換しません。
