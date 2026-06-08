# モデル実装メモ

全モデルは共通して `scaled ΔT` を返します。

```text
入力: state_hist, control_hist, control_next
出力: target_delta と同じ shape の ΔT
```

Neural ODE / PINN / DeepONet / FNO はこの実装には含めていません。今回の目的は少数測定点の温度を固定時間ステップで逐次更新することなので、まずは以下の実装で比較します。

## linear_rc

```text
ΔT = W [T_cur, u_next] + b
```

最低限の線形baselineです。これに勝てない場合は、データ分割、正規化、学習率、過学習を確認します。

## mlp

```text
ΔT = MLP(flatten(T_hist, u_hist), u_next)
```

小規模データで最初に見る深層baselineです。

## gru

```text
h = GRU([T_hist, u_hist])
ΔT = Head(h_last, u_next)
```

複数時定数や操作履歴が効く場合の候補です。

## tcn

```text
z = Conv1D([T_hist, u_hist])
ΔT = Head(z_last, u_next)
```

固定dt、固定履歴長に向いた畳み込み時系列モデルです。

## thermal_state_space

```text
T_next = T_eq(u) + a(u) * (T_cur - T_eq(u)) + residual(T_cur, u)
ΔT = T_next - T_cur
```

平衡温度と安定減衰を分けるため、rollout安定性を見たい本問題の第一候補です。

## graph_rc

```text
ΔT_i = s_g * Σ_j G_ij (T_j - T_i) / M_i
     + s_q * source_prior_i(u) / M_i
     + source_nn_i(u)
     + residual_i(T,u)
```

`configs/cell_nodes.csv` と `configs/cell_edges.csv` から、Cell位置関係・熱抵抗priorを読みます。少数センサー向けに、外部GNNライブラリは使わず、明示的な行列演算だけで実装しています。

学習後は以下を確認します。

```text
graph/learned_conductance.csv
graph/graph_diagnostics.csv
graph/source_weight.csv
plots/graph_conductance_prior.png
plots/graph_conductance_learned.png
```

`learned_over_prior` が極端に大きい/小さいedgeは、熱抵抗表とCAE条件のズレを示す可能性があります。

## optional rollout loss

デフォルトは通常のone-step ΔT lossです。

```yaml
train:
  rollout_loss_weight: 0.0
```

rolloutで誤差が蓄積する場合のみ、軽く有効化します。

```yaml
train:
  rollout_loss_weight: 0.2
  rollout_loss_steps: 3
```

毎epochの本格rollout評価は重くなるため、学習中は短いfuture列だけを使い、最終評価で実rolloutを確認します。


## dynamic controls / monitoring update

この版では、Neural ODE / PINN / DeepONet / FNO を追加せず、既存モデルを実利用側に寄せています。追加した主な実装は以下です。

- `data.header: false/true/auto` によるヘッダーあり/なしCSV読込
- CAE CSV内の時間変化 `brine/heater/plasma` 列の学習利用
- `schedule_interpolation: previous/linear`
- `features.use_effective_controls` と `control_lag_tau` による一次遅れ有効入力
- `prediction.mode: monitor` による実測ログの一ステップ残差監視
- monitor logのリサンプリング
- `model.graph.ignore_unknown_edges` による3点サブグラフ対応
- `train.graph_prior_loss_weight` によるGraph-RC熱抵抗prior正則化

重要な注意点として、定数条件CAEだけで学習したモデルに時間変化scheduleを入れることは、コード上は可能でも物理的には外挿になりやすいです。時間変化ヒーター・ブラインを実用する場合は、step/ramp/pulse/recipe形状のCAEを学習データへ追加してください。
