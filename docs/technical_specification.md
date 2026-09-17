# thermal-cell-practical 技術仕様書

## 0. 文書情報

| 項目 | 内容 |
|---|---|
| 文書の目的 | 熱RCモデルの同定、温度予測、状態監視に関する実装仕様と適用条件を定める |
| 文書版 | 1.1 |
| 基準日 | 2026-09-16（Asia/Tokyo） |
| 対象パッケージ | thermal-cell-practical 1.0.0 |
| 対象コミット | f1b979f9b57ae2262a74cf26d6ba6b250f7698ab |
| 主な対象 | <code>src/celltemp</code>、公開CLI 3系統、TopCell、COMSOL評価系 |
| 実装規模 | 中核Python 26ファイル、5,044行、テスト関数116件 |
| 記述形式 | Markdown、LaTeX数式、Mermaid図 |

### 0.1 改訂履歴

| 版 | 日付 | 改訂内容 |
|---|---|---|
| 1.0 | 2026-09-15 | 初版 |
| 1.1 | 2026-09-16 | 物理妥当性、統計設計、検証境界を中心に構成と記述を全面改訂 |

### 0.2 本書の読み方

本書では、コードが実際に行う処理を「実装仕様」、装置ごとに決める事項を「設計入力」、特定データで得た数値を「評価結果」と呼ぶ。要求の強さは次のように区別する。

| 表現 | 意味 |
|---|---|
| 必須 | 満たさない入力または成果物は受け付けない |
| 推奨 | 適用妥当性や再現性を確保するために実施する |
| 任意 | 対象設備や運用目的に応じて選択する |

仕様の優先順位は、対象コミットのソースコード、機械可読なYAML/JSON/CSV、テスト、説明文書の順とする。評価値は特定のデータ、乱数seed、閾値に対する結果であり、一般的な精度保証ではない。

---

## 1. 要旨

本パッケージは、少数の熱ノードで構成した集中定数熱回路を用いて、過渡温度データから物理係数を同定し、同じモデルで将来温度の予測とオンライン状態監視を行うための基盤である。学習器はブラックボックス回帰ではなく、熱収支式を時間積分した軌道と観測軌道の差を最小化する。熱容量、熱コンダクタンス、熱源、境界温度、アクチュエータ遅れは、SI単位に対応する量として扱う [L1][L4]。

設計上の要点は次の3点である。

1. **モデルを共通化する。** 同定、予測、監視が同じ <code>ThermalRCModel</code> とアクチュエータ状態を使用するため、処理系ごとのモデル差を生じさせない。
2. **因果性を評価条件に組み込む。** 学習時のモデル選択は初期観測だけからの開ループ誤差、予測は観測履歴の終端からの開ループ誤差で評価し、将来温度の混入を禁止する。
3. **適用限界を数値結果と分けて管理する。** 合成ベンチマークと線形化COMSOLデータでは良好な結果を得ている一方、非線形・高忠実度COMSOL評価はメッシュおよび実験の資格が未達であり、用途をモデル構造のスクリーニングに限定する。

現在の主な評価結果を次表に示す。完全精度は各参照先のJSONを正とする。

| 評価系 | 対象 | 主結果 | 解釈 |
|---|---|---:|---|
| TopCell [E1] | 未学習の合成予測11ケース | 平均RMSE 0.143 K、最大1.303 K | 実装回帰と既知真値での確認 |
| 線形COMSOL [E2] | 外部予測8ケース | 平均RMSE 0.036 K、最大0.083 K | 線形化した体積平均温度に対するCAE比較 |
| 高忠実度COMSOL [E4] | HV01/HV02 | 平均RMSE 0.096 K | 隣接メッシュ差0.496 Kの範囲内で、厳密妥当化は未達 |

### 1.1 想定読者

| 読者 | 本書で確認する事項 |
|---|---|
| 物理・熱設計担当 | 熱収支、符号、単位、ノード分割、境界条件、保存則、モデル縮約誤差 |
| データサイエンティスト | データ粒度、分割漏洩、欠測、損失関数、識別可能性、評価指標、不確かさ |
| ソフトウェア担当 | モジュール責務、入出力契約、成果物schema、例外条件、品質ゲート |
| 運用・品質保証担当 | 適用範囲、監視統計、再現性、リリース条件、未完了の妥当化項目 |

### 1.2 適用範囲

| 用途 | 現状の位置付け | 前提 |
|---|---|---|
| 熱係数・時定数の軌道同定 | 適用可能 | 入力励振、ノード構造、熱容量の事前設計が妥当 |
| 学習域内の温度軌道予測 | 適用可能 | 予測起点の状態推定と範囲判定を実施 |
| 未学習入力波形の比較 | 条件付き | 入力範囲、時間刻み、予測時間、変化率を確認 |
| 未指令発熱・相対センサずれの監視 | 条件付き | 観測可能性、ノイズ、警報閾値を対象設備で調整 |
| CAE反復の低次代替 | スクリーニング用途 | 独立CAEまたは実験で誤差を管理 |
| 局所最高温度の安全判定 | 対象外 | 本モデルの状態は代表温度または平均温度 |
| 安全保護系の単独モデル | 対象外 | 独立したフェイルセーフ機構が必要 |

### 1.3 解決する課題

3次元CAEは空間温度、流れ、放射を詳細に表現できるが、反復計算とオンライン利用の費用が大きい。一方、純粋なデータ駆動モデルは高速化しやすいものの、少量データ、外挿、単位整合、保存則、故障時の説明性が課題となる。本パッケージは両者の中間に位置し、装置の主要な熱時定数を少数ノードへ縮約して、物理制約を保ったまま係数をデータから調整する。

---

## 2. システム構成

### 2.1 システム境界

~~~mermaid
flowchart LR
    A[CAE・実験・ロガーCSV] --> B[時系列アダプタ]
    S[system.yaml<br/>熱ノード・経路・センサ] --> C[熱系仕様]
    G[config.yaml] --> D{公開CLI}
    B --> D
    C --> D
    D -->|train| E[物理パラメータ同定]
    E --> F[モデル成果物]
    F -->|forecast| H[履歴同化と開ループ予測]
    F -->|monitor| I[因果状態監視]
    H --> J[予測値・状態分散・範囲判定]
    I --> K[状態・未知熱・バイアス・NIS]
    X[TopCell・COMSOL評価系] --> A
    J --> X
    K --> X
~~~

公開CLIは <code>train</code>、<code>forecast</code>、<code>monitor</code> の3系統である。COMSOLモデルの生成、メッシュ評価、合否判定は <code>external_tools</code> に分離し、中核パッケージへCOMSOL依存を持ち込まない。

### 2.2 依存方向

~~~text
cli
  → workflows
    → artifact / learning / inference
      → engine / io
        → domain / config
~~~

<code>.importlinter</code> が上記の一方向依存を検査する。<code>domain</code> はPyTorch、pandas、YAMLへ依存せず、数値計算層はpandasとYAMLを直接扱わない。この分離により、物理計算、表形式入出力、運用処理を個別に試験できる [L3]。

### 2.3 用語

| 用語 | 定義 |
|---|---|
| 熱ノード | 一様温度で近似する集中熱容量 |
| 指令値 $u$ | CSVに記録された操作量 |
| 実効操作量 $a$ | 一次遅れ後に熱モデルへ作用する量 |
| 観測値 $y$ | センサ温度。直接ノード温度または加重平均 |
| モデル成果物 | <code>model.pt</code>、<code>system.yaml</code>、<code>metadata.json</code> の組 |
| 条件付き誤差 | 同じ軌道の将来観測を用いて隠れ初期温度を推定した誤差 |
| 因果誤差 | 初期時点までの観測だけを用いた開ループ誤差 |
| 未知熱 | 指令値で説明されない、既知空間基底上の符号付き発熱量 |
| センサバイアス | 選択した基準条件の下で識別できる相対的な温度ずれ |
| NIS | 観測残差をその共分散で規格化した二次形式 |

### 2.4 中核モジュールの責務

| 責務 | 主ファイル | 入力 | 出力・状態 | 主な拒否条件 |
|---|---|---|---|---|
| CLI振り分け | <code>cli.py</code> | コマンド、設定、上書き値 | 各処理系の呼出し | 未知コマンド、設定欠落 |
| 設定処理 | <code>config.py</code> | YAML、dot形式上書き | 検証済み辞書、基準パス | 未知キー、非mapping |
| 時系列領域モデル | <code>domain/trajectory.py</code> | 時刻、温度、指令値、mask | 不変 <code>Trajectory</code> | 非単調時刻、shape不一致 |
| 熱系領域モデル | <code>domain/topology.py</code> | ノード、熱経路、センサ | <code>ThermalSystemSpec</code> | 非正容量、重複経路、未知参照 |
| CSV変換 | <code>io/trajectory.py</code> | CSV/DataFrame | ケース別時系列 | 必須列欠落、初期観測なし |
| system YAML変換 | <code>io/system.py</code> | schema version 3 | 熱系仕様 | 未知キー、型、参照違反 |
| 応答則 | <code>engine/laws.py</code> | 応答則、実効操作量 | 熱源・コンダクタンス | 操作量次元不一致 |
| 時間積分 | <code>engine/integrator.py</code> | $A,b,x,\Delta t$ | 1区間後の状態 | shape、積分条件違反 |
| 熱RC計算 | <code>engine/rc.py</code> | 熱系仕様、指令値、状態 | 温度、実効操作量、観測 | 積分器非対応の応答則 |
| 状態推定 | <code>engine/observer.py</code> | モデル、観測、雑音設定 | 事後状態、共分散、残差 | 不正標準偏差、未知基準センサ |
| データ分割 | <code>learning/split.py</code> | 全ケース、seed、比率または表 | train/val/test | 表の欠落・余分なID |
| 学習目的関数 | <code>learning/objective.py</code> | モデル、時系列、mask | 軌道、Huber損失、RMSE | 観測のない評価区間 |
| 最適化 | <code>learning/trainer.py</code> | モデル、ケース、設定 | 最良状態、学習履歴 | 非有限損失・勾配 |
| 配備API | <code>inference.py</code> | モデル、時系列、観測器 | 予測結果、監視結果 | 予測境界、layout不一致 |
| 成果物管理 | <code>artifact.py</code> | モデル、metadata | schema version 4 | hash、buffer、dtype不一致 |
| 共通運用処理 | <code>workflows/common.py</code> | パス、設定、入力 | staging、manifest | 危険な出力先、上書き禁止 |

### 2.5 外部評価系の責務

| 評価系 | 主な責務 | 中核との境界 |
|---|---|---|
| TopCell | 合成真値、未学習入力、欠測、故障、モデル構造差を生成 | 公開CSVだけを中核へ渡す |
| 線形COMSOL | 固体伝導、接触、一定対流の温度軌道を生成 | 自己完結CSVへ正規化 |
| 非線形COMSOL | 共役熱流動、流速、放射の影響を評価 | モデル構造差の評価に限定 |
| メッシュ評価 | global/local meshのQoI差を比較 | strict qualificationとbenchmark利用を分離 |
| 高忠実度評価 | HV01/HV02の対応過渡を比較 | 実測未提供を妥当化済みと数えない |
| CAE報告生成 | 正規化済みCSV/JSONから報告を構築 | 数値の正本をHTMLへ置かない |

---

## 3. データおよび設定の契約

### 3.1 データ粒度

学習、予測、監視の入力は共通の自己完結CSV形式とする。

| 項目 | 規約 |
|---|---|
| ファイル粒度 | 1ファイル＝1運転ケース |
| 行粒度 | 1行＝1ケースの1時刻 |
| ケースID | ファイル名のstem。入力ディレクトリ内で一意 |
| 主キー | 実質的に <code>(case_id, time)</code> |
| センサ順 | <code>system.yaml</code> に記載した順序 |
| 操作量順 | <code>system.yaml</code> に記載した順序 |
| 欠測 | 空欄またはNaN。無限大は不可 |

<code>Trajectory</code> は生成時に配列を複製し、read-onlyとする。生成後に呼出し元の配列を変更しても、時系列の不変条件は変化しない。

### 3.2 標準時系列

| 量 | shape | 条件 | 意味 |
|---|---:|---|---|
| <code>time</code> | $[N]$ | float64、有限、厳密単調増加、$N\ge2$ | 観測時刻 |
| <code>temperature</code> | $[N,p]$ | 観測位置は有限 | センサ温度 |
| <code>commands</code> | $[N-1,m]$ | 全要素有限 | 各区間の指令値 |
| <code>observation_mask</code> | $[N,p]$ | bool | 有効観測位置 |
| <code>initial_actuator</code> | $[m]$ またはnull | 有限 | 最初の区間に入る直前の実効操作量 |
| <code>sensor_names</code> | $p$ 個 | 一意、順序付き | 観測行列の行順 |
| <code>control_names</code> | $m$ 個 | 一意、順序付き | 操作量の列順 |

### 3.3 時刻とゼロ次ホールド

内部表現では、指令値 <code>commands[k]</code> を区間 $[t_k,t_{k+1})$ に適用する。

~~~text
行 k                                             行 k+1
T[k], y[k]  ├───────────────────────────────────┤ T[k+1], y[k+1]
             commands[k] をゼロ次ホールド
             a[k] → a[k+1]
             熱状態を時間積分
                                                  監視では y[k+1] で更新
~~~

CSVは通常、全時刻行に操作量を持つ。<code>control_convention: left</code> ではCSV行 $k$、<code>right</code> ではCSV行 $k+1$ を内部区間へ割り当てる。したがって、leftでは末尾行、rightでは先頭行の操作量を遷移に用いない。装置ログの時刻ラベルとこの規約がずれると、熱源ゲインと時定数の双方に系統誤差が入る。

### 3.4 CSV列

| 列 | 必須性 | 規約 |
|---|---|---|
| <code>time</code> | 必須 | 列名は設定変更可。有限、厳密増加 |
| 各センサ名 | 必須 | 欠測値は許容条件に従う |
| 各操作量名 | 必須 | 全行有限 |
| <code>initial_effective_&lt;control&gt;</code> | 任意 | 先頭行のみ使用。欠けた操作量は既定値を使用 |
| 評価用真値、補助列 | 任意 | 中核ローダは使用しない |

| 処理系 | 観測条件 |
|---|---|
| 学習 | 先頭行と、それ以後に少なくとも1観測。欠測利用は設定で明示 |
| 予測 | 観測行の連続prefixに続いて、全センサ未観測のfuture suffixを置く |
| 監視 | 先頭行に少なくとも1観測。以後はセンサ別または全センサ欠測を許容 |

### 3.5 単位

| 物理量 | 単位 | 実装上の扱い |
|---|---|---|
| 時刻、$\Delta t$、時定数 | s | $\Delta t>0$ |
| 温度 | °CまたはK | プロジェクト内で一貫。温度差の単位はK |
| 熱容量 | J/K | 正値、現行学習では固定 |
| 内部・境界コンダクタンス | W/K | 非負 |
| 指令熱源 | W | 応答則の出力は非負 |
| 未知熱 | W | 観測器の空間基底上で符号付き |
| センサバイアス | K | 指定した基準条件に対する値 |
| Huber閾値、温度標準偏差 | K | 温度誤差の尺度 |
| 未知熱process std | W/$\sqrt{\mathrm{s}}$ | 連続時間ランダムウォーク強度 |
| バイアスprocess std | K/$\sqrt{\mathrm{s}}$ | 連続時間ランダムウォーク強度 |

単位変換は自動で行わない。例えば、kWをWへ、分を秒へ変換する責務はデータ生成側にある。値域検査は単位誤りの一部を検出できるが、物理単位そのものを判定するものではない [L2]。

### 3.6 欠測とデータ品質

損失関数と状態更新はmaskが真の要素だけを使用する。欠測値を0へ置換しない。ただし、欠測を無視して計算できることと、評価の偏りがないことは別問題である。高温時や故障時ほど欠測しやすい場合、観測済みデータだけのRMSEは実運用誤差を過小評価する可能性がある。

| 検査 | 実装 | 運用で追加する検査 |
|---|---|---|
| 時刻逆行・重複 | 厳密増加条件で拒否 | logger間の時刻同期 |
| 必須列 | 欠落を拒否 | 単位・校正履歴 |
| 非有限値 | 指令値の非有限と温度の無限大を拒否 | 飽和値、欠測コード |
| 一定刻み | <code>data.dt</code> 指定時に照合 | clock drift、jitter分布 |
| 温度範囲 | <code>temp_min/max</code> で照合 | 装置モード別範囲 |
| 欠測率 | maskとして保持 | センサ別・運転相別の欠測率 |
| ケース重複 | ファイルstem一意 | 内容hash、再計算variant |

### 3.7 <code>system.yaml</code> schema version 3

| section | 主フィールド | 制約 |
|---|---|---|
| <code>nodes</code> | name、heat_capacity | 非空、一意、熱容量 $>0$ |
| <code>actuators</code> | name、tau、learnable | $\tau\ge0$。$\tau=0$ は学習不可 |
| <code>edges</code> | nodes[2]、conductance | 異なる既知ノード、無向重複なし |
| <code>sources</code> | name、node_weights、heat_rate | weight非負、少なくとも1要素が正 |
| <code>boundaries</code> | name、node_weights、reservoir_temperature、conductance | reservoirは定数または操作量の一次式 |
| <code>sensors</code> | nameとnode、またはnode_weights | weight非負、行和1 |

熱源と境界の <code>node_weights</code> は熱量または熱伝達の分配係数であり、和を1に限定しない。センサweightだけが温度平均を表すため、行和1を要求する。

### 3.8 主要設定

| key | 既定値 | 役割 |
|---|---:|---|
| <code>seed</code> | 42 | データ分割、NumPy、PyTorch |
| <code>engine.integrator</code> | exact | exactまたはimplicit |
| <code>data.control_convention</code> | left | CSV操作量の時刻規約 |
| <code>data.dt</code> | null | 数値指定時は期待一定刻み |
| <code>data.allow_missing_temperatures</code> | false | 学習データの欠測許可 |
| <code>split.method</code> | random | randomまたはexplicit |
| <code>split.train_ratio</code> | 0.70 | recipe group単位 |
| <code>split.val_ratio</code> | 0.15 | test分を残す |
| <code>project.overwrite_run</code> | false | 既存学習runの置換 |
| <code>forecast/monitor.device</code> | cpu | モデル実行device |

相対パスは実行時のカレントディレクトリではなく、設定ファイルの配置ディレクトリを基準に解決する。

---

## 4. 物理モデル

### 4.1 記号

| 記号 | 次元・単位 | 定義 |
|---|---|---|
| $n,p,m$ | 個数 | 熱ノード数、センサ数、操作量数 |
| $T\in\mathbb{R}^n$ | Kまたは°C | ノード温度 |
| $u,a\in\mathbb{R}^m$ | 装置固有 | 指令値、実効操作量 |
| $C=\operatorname{diag}(C_i)$ | J/K | 熱容量行列 |
| $B_e$ | edge×node | 符号付き接続行列 |
| $g(a)$ | W/K | 内部コンダクタンス |
| $L(a)$ | W/K | 重み付きグラフLaplacian |
| $H$ | sensor×node | 観測行列 |
| $W_d$ | disturbance×node | 未知熱の空間基底 |

### 4.2 ノード熱収支

各ノードは内部で温度が一様な集中熱容量とみなす。ノード $i$ の第一法則は

$$
C_i\frac{dT_i}{dt}
=
\sum_{j\in\mathcal N(i)}g_{ij}(a)(T_j-T_i)
+
\sum_s q_s(a)w_{s,i}
+
\sum_b h_b(a)w_{b,i}\left(T_{r,b}(a)-T_i\right)
+
\sum_\ell W_{d,\ell i}d_\ell .
$$

内部熱流は共有コンダクタンスを用い、

$$
q_{i\leftarrow j}=g_{ij}(a)(T_j-T_i),
\qquad
q_{j\leftarrow i}=-q_{i\leftarrow j}
$$

とする。正の値は対象ノードへの流入を表す。これにより、内部経路は相反性を持ち、閉じた系で内部熱流の総和が0になる。

### 4.3 行列表現

接続行列を用いると、

$$
L(a)=B_e^\mathsf{T}\operatorname{diag}(g(a))B_e
$$

である。連続時間の熱収支は

$$
C\dot T=
-L(a)T
+\sum_s q_s(a)w_s
+\sum_b h_b(a)\operatorname{diag}(w_b)
\left(T_{r,b}(a)\mathbf 1-T\right)
+W_d^\mathsf{T}d .
$$

熱源のweightは正規化を必須としないため、熱源 $s$ の全ノード合計入熱は

$$
Q_{s,\mathrm{total}}=q_s(a)\mathbf 1^\mathsf{T}w_s
$$

となる。公称モデルでは未知熱項を除き、

$$
\dot T=A(a)T+b(a)
$$

$$
A(a)=
-C^{-1}
\left[
L(a)+\sum_b h_b(a)\operatorname{diag}(w_b)
\right]
$$

$$
b(a)=
C^{-1}
\left[
\sum_s q_s(a)w_s+
\sum_b h_b(a)w_bT_{r,b}(a)
\right].
$$

この電気―熱アナロジーは、電子機器を含む集中定数熱回路で一般に用いられる [R1][R2]。

### 4.4 操作量に対する応答則

内部コンダクタンス、境界コンダクタンス、熱源強度には共通の応答則を用いる。

| 応答則 | 式 | 用途と制約 |
|---|---|---|
| constant | $\lambda(u)=v$ | $v>0$ |
| positive_part | $\lambda(u)=k\max(u-\theta,0)$ | $k>0$、出力非負 |
| power_law | $\lambda(u)=c_0+c_1(\max(u,0)/u_{\rm ref})^\alpha$ | $c_0,c_1\ge0$、少なくとも一方が正、$\alpha>0$ |

<code>power_law</code> は $u=0$ のべき乗成分を0とし、出力はoffset $c_0$ となる。threshold、操作量名、referenceは固定し、正の係数だけを対数パラメータ化して学習する。

### 4.5 境界温度

境界reservoir温度は

$$
T_{r,b}(a)=\gamma_{0,b}+\gamma_{1,b}a_{j(b)}
$$

とする。境界コンダクタンスを決める操作量とreservoir温度を決める操作量は別に指定できる。例えば、流量で熱伝達率を変え、入口温度でreservoir温度を変える構成を表現できる。

### 4.6 アクチュエータ

各実効操作量は一次遅れ

$$
\tau_j\dot a_j=u_j-a_j
$$

に従う。区間内で $u_j$ が一定なら、

$$
a_j(t+\Delta t)
=u_j+\left(a_j(t)-u_j\right)\exp\left(-\frac{\Delta t}{\tau_j}\right).
$$

$\tau_j=0$ は遅れなしとして $a_j=u_j$ とする。初期実効操作量をCSVで与えない場合は最初の区間指令値を使用する。開始前に装置が整定していないデータでは、この既定値を使わず <code>initial_effective_*</code> を与える。

### 4.7 観測モデル

センサ観測は

$$
y=HT+v
$$

で表す。単一ノードセンサは $H$ の1要素が1、面積平均・体積平均に相当するセンサは非負weightの行和を1とする。したがって、センサ温度はノード温度の凸結合となる。

### 4.8 物理的不変条件

| 性質 | 成立条件 | 根拠 |
|---|---|---|
| 内部熱流の相反性 | 無向edgeが1つの $g_{ij}$ を共有 | $q_{i\leftarrow j}=-q_{j\leftarrow i}$ |
| 閉じた系のエネルギー保存 | 熱源・境界なし | $\frac{d}{dt}\mathbf1^\mathsf{T}CT=0$ |
| 温度差の散逸 | 熱源・境界なし | $T^\mathsf{T}LT=\sum_e g_e(T_i-T_j)^2\ge0$ |
| 正の時間発展 | $C_i>0,\ g_e\ge0$ | $-C^{-1}L$ はMetzler行列 |
| 閉じた系の最大値原理 | 上記条件 | $e^{At}$ が非負で定数温度を保存 |
| 学習中の受動性 | 正値係数を対数表現 | 最適化で負のコンダクタンスを生成しない |

正システムとMetzler行列の一般論は [R3] を参照する。異なる温度の境界または熱源がある場合、初期温度の単純な上下限は適用できない。

### 4.9 集中定数化の妥当性

1つの熱ノードにまとめた領域は、領域内部の温度均一化が外部との熱交換より十分速い必要がある。代表的な確認量はBiot数

$$
\operatorname{Bi}_i=\frac{h_{\mathrm{eff},i}L_{c,i}}{k_i}
$$

である。Biot数が小さいほど単一温度近似に適する。ただし、接触抵抗、局所熱源、複数境界を持つ装置では、単一のBiot数だけで採否を決めず、領域内温度差と要求精度をCAEまたは実測で確認する。

熱系の支配時定数は、固定した操作点で

$$
K\phi_r=\lambda_r C\phi_r,\qquad \lambda_r>0,
\qquad
K=L+\sum_b h_b\operatorname{diag}(w_b),
\qquad
\tau_r=\lambda_r^{-1}
$$

と評価できる。対象時間帯で寄与の大きい固有modeを再現できるようにノードを分ける。次のいずれかに該当する場合はノード追加または高忠実度モデルが必要となる。

| 判定項目 | 見直しの兆候 |
|---|---|
| 領域内温度差 | 要求誤差または安全余裕に対して無視できない |
| 残差の空間構造 | 特定センサだけに位相遅れや定常偏差が残る |
| 残差の運転依存 | 高温、低流量、特定接触条件だけで増える |
| 固有時定数 | 実測に複数の未再現時定数が見える |
| hotspot | 平均温度との差が設計判断を変える |

### 4.10 現行モデルに含まれない現象

- 放射の $T^4$ 依存
- 温度依存の熱容量、熱伝導率、接触抵抗
- 相変化、潜熱、ヒステリシス
- dead time、多段アクチュエータ、overshoot
- 流れ場、圧力損失、自然対流・乱流遷移の内部状態
- 接触の開閉や運転中のtopology変更
- 3次元局所最高温度

これらの影響が小さいことは、学習誤差ではなく、対象運転域の独立データで確認する。

---

## 5. 数値計算

### 5.1 厳密affine積分

熱係数とpositive-part熱源のactive setが区間内で固定される場合、温度とアクチュエータを結合した状態

$$
x=\begin{bmatrix}T\\a\end{bmatrix}
$$

は

$$
\dot x=Fx+c
$$

となる。実装は拡大行列の指数関数

$$
\begin{bmatrix}x_{k+1}\\1\end{bmatrix}
=
\exp\left(
\begin{bmatrix}F&c\\0&0\end{bmatrix}\Delta t_k
\right)
\begin{bmatrix}x_k\\1\end{bmatrix}
$$

を計算する。$F^{-1}$ を構成しないため、閉じた熱系のゼロ固有値や遅れなしアクチュエータを含む特異系にも適用できる [R5][R6][R7]。行列指数は <code>torch.linalg.matrix_exp</code> を用いる [R12]。

同じ演算を

$$
x_{k+1}=\Phi_kx_k+\Gamma_kc_k,
\quad
\Phi_k=e^{F_k\Delta t_k},
\quad
\Gamma_k=\int_0^{\Delta t_k}e^{F_ks}\,ds
$$

と表すこともできる。batch計算では、同一の係数行列と時間刻みに対する $\Phi_k,\Gamma_k$ を再利用する。

### 5.2 threshold通過

一次遅れの実効操作量がpositive-partのthreshold $\theta$ を区間内で横切る場合、交差時刻

$$
t_c=-\tau\log\left(\frac{\theta-u}{a_0-u}\right)
$$

を解析的に求める。$0<t_c<\Delta t$ の交差だけを採用し、区間をactive setごとに分割して厳密affine積分を行う。これにより、区間中央のon/off判定だけに依存する誤差を避ける。

### 5.3 exact積分器の適用条件

| 経路 | 操作量の時定数 | exact |
|---|---:|---|
| 定数の内部・境界コンダクタンス、熱源 | 任意 | 可 |
| 操作量依存の内部コンダクタンス | 0のみ | 可 |
| 操作量依存の境界コンダクタンス | 0のみ | 可 |
| positive-part熱源 | 0または正 | 可。正の場合はthresholdで分割 |
| power-law熱源 | 0のみ | 可 |
| 操作量に一次依存するreservoir温度 | 0または正 | 可 |

遅れを持つ操作量がコンダクタンスまたはpower-law熱源を連続的に変える場合、区間内の系はaffineでなくなる。その構成でexactを選ぶとmodel構築時に拒否し、implicitを要求する。

### 5.4 陰的Euler積分

implicitでは、アクチュエータの解析的な区間中央値 $a_{k+1/2}$ で $A,b$ を評価し、

$$
T_{k+1}
=
\left(I-\Delta t_kA(a_{k+1/2})\right)^{-1}
\left[T_k+\Delta t_kb(a_{k+1/2})\right]
$$

を <code>torch.linalg.solve</code> で解く。Backward EulerはA-stableな一次法であり、stiffな受動熱系に対する代替積分法となる [R8]。操作量依存係数の区間内変化は中央値近似であるため、exactと同じ意味での厳密解ではない。

### 5.5 積分器の選択

| 判断項目 | exact | implicit |
|---|---|---|
| 区間affine条件 | 必須 | 不要 |
| 時間離散化誤差 | 仮定内ではなし | 一次精度 |
| stiff系 | 行列指数で安定 | A-stable |
| 主演算 | 密行列指数 | 密線形方程式 |
| 適する用途 | 少数ノード、ゼロ次ホールド | 遅れた非線形係数 |

### 5.6 計算量とcache

熱状態次元を $q=n+m$、観測器状態次元を $q_o=n+r+p_b$ とすると、異なる演算子1つ当たりの密行列指数または線形solveは概ね $O(q^3)$ または $O(q_o^3)$ である。時系列状態の保存量はケース当たり $O(Nn)$ となる。

| 最適化 | 実装 |
|---|---|
| 学習batch | 同じ長さの軌道をまとめる |
| 可変刻み | $\Delta t$ を <code>[batch, step]</code> で保持 |
| exact演算子 | 時間刻み、active set、係数を変える操作量が同じ場合に再利用 |
| 観測器演算子 | $(\Delta t,F)$ をkeyとする最大256件のLRU cache |

本実装は少数ノードを対象とし、数百から数千自由度の疎行列CAE solverを置き換える設計ではない。

### 5.7 数値異常の扱い

- 既定dtypeはfloat64。成果物の読込みはfloat32またはfloat64に限定する。
- 逆行列を明示せず、行列指数またはlinear solveを用いる。
- 非有限の状態、損失、勾配、設定値を拒否する。
- 勾配normを設定値でclipする。
- 学習するlog multiplierを $[-4,4]$ に制限する。
- 共分散は演算後に $(P+P^\mathsf{T})/2$ で対称化する。

---

## 6. 物理パラメータ同定

### 6.1 学習問題の定義

PyTorchはニューラルネットワークを構成するためではなく、物理軌道の自動微分と最適化に使用する [R10][R11][R12]。学習対象は事前に定義した熱回路の少数パラメータであり、topologyや熱容量をデータから自動探索しない。

| 量 | 学習 | 理由 |
|---|---|---|
| 内部コンダクタンス | YAMLで指定 | 熱結合を軌道から調整 |
| 境界コンダクタンス | YAMLで指定 | 対流・接触の有効値を調整 |
| 熱源gain/scale/offset | YAMLで指定 | 入力―発熱変換を調整 |
| power-law exponent | YAMLで指定 | 流量等への非線形応答 |
| アクチュエータ時定数 | YAMLで指定 | 指令から実効作用までの遅れ |
| 熱容量 | 固定 | 全熱流量とのglobal scale非識別性を避ける |
| ケース別初期温度 | 毎軌道で推定 | nuisance量として扱い、成果物へ保存しない |
| topology、weight、センサ配置 | 固定 | 物理設計入力とする |

### 6.2 正値パラメータ化

学習する正値係数 $p$ は、工学的事前値 $p_0>0$ とlog multiplier $\ell$ により

$$
p=p_0e^\ell
$$

とする。$\ell=0$ は事前値に一致し、最適化中も $p>0$ を保つ。実装上の範囲は

$$
p_0e^{-4}\le p\le p_0e^4
$$

である。この制約は負のコンダクタンスや負の時定数を防ぐが、入力励振不足によるパラメータ相関は解消しない。

### 6.3 ケース分割と漏洩防止

random splitは行ではなく軌道全体を単位とする。さらに、指令値配列と $\Delta t$ 配列が

$$
\operatorname{allclose}
\left(x_1,x_2;\ r_{\mathrm{tol}}=10^{-9},
a_{\mathrm{tol}}=10^{-9}\right)
$$

で一致するケースを同じrecipe groupへまとめ、group単位でtrain、validation、testへ割り当てる。同じ指令履歴で初期温度やノイズだけが異なる軌道が複数partitionへ入ることを防ぐ。

<code>split.method=explicit</code> では全ケースIDを表で一度ずつ指定する。実装は表の網羅性を検査するが、同じrecipeが異なるpartitionへ割り当てられていないかは利用者が確認する。

| 分割上の要件 | 目的 |
|---|---|
| 軌道を分断しない | 未来行の漏洩防止 |
| 同一recipeを同じpartitionへ置く | 再計算variantの漏洩防止 |
| validationを独立に確保する | model selectionの独立性 |
| 最終評価ケースを学習探索先から分離する | 汎化性能の評価 |

### 6.4 隠れ初期温度

観測済み初期センサに対応する行列を $H_0$ とする。基準初期温度 $\bar T_0$ は、観測センサの平均温度を未観測ノードの弱い事前値として、ridge付き線形方程式から求める。

SVD

$$
H_0=U\Sigma V^\mathsf{T}
$$

からnull spaceの基底 $N$ を取り、

$$
T_0(z)=\bar T_0+Nz,\qquad H_0Nz=0
$$

と置く。これにより、隠れ初期温度の補正が、基準初期温度の初期予測観測を変えない。基準値自体は $10^{-8}$ のridgeを含むため、重複・加重平均センサに対する測定値との厳密一致を前提としない。

熱RC軌道は初期温度に対してaffineである。$\bar T_0$ と各 $\bar T_0+N_{\cdot j}$ をbatch計算し、観測軌道に対する感度行列 $J$ を求める。欠測を除いた残差 $r$ に対し、

$$
\left[
J^\mathsf{T}J+
\left(\epsilon_{\mathrm{num}}+\sigma_0^{-2}\right)I
\right]\hat z
=J^\mathsf{T}r
$$

$$
\epsilon_{\mathrm{num}}
=
\sqrt{\epsilon_{\mathrm{machine}}}\,
\max\left(
1,\frac{\operatorname{tr}(J^\mathsf{T}J)}{\dim z}
\right)
$$

を解く。$\sigma_0$ は <code>initial_temperature_prior_std</code> である。推定した $T_0$ はケース固有量であり、共有モデル成果物に含めない。

### 6.5 損失関数

温度残差 $e=\hat y-y$ にはHuber損失 [R9][R13] を用いる。

$$
\rho_\delta(e)=
\begin{cases}
\frac12e^2,& |e|\le\delta,\\
\delta\left(|e|-\frac12\delta\right),& |e|>\delta.
\end{cases}
$$

既定値は $\delta=1$ Kである。ケース $i$ の有効評価点を $\Omega_i$ とすると、

$$
\mathcal L_i=
\frac1{|\Omega_i|}
\sum_{(k,s)\in\Omega_i}
\rho_\delta(\hat y_{i,k,s}-y_{i,k,s})
$$

$$
\mathcal L=
\frac1B\sum_{i=1}^{B}\mathcal L_i
+
\lambda_{\mathrm{prior}}
\frac1M\sum_{j=1}^{M}\ell_j^2 .
$$

最初の式をケース内平均、次にケース間平均とするため、長い軌道や観測点の多いケースだけが損失を支配しない。初期化に直接使う先頭行は通常の学習・評価maskから除く。

### 6.6 最適化

~~~mermaid
flowchart TD
    A[全ケース] --> B[recipe単位で分割]
    B --> C[事前値から熱RCモデルを構築]
    C --> D[epoch開始]
    D --> E[各ケースから全軌道またはwindowを選択]
    E --> F[SVDで隠れ初期温度を推定]
    F --> G[軌道長ごとにbatch積分]
    G --> H[mask付きHuber損失をケース別に計算]
    H --> I[ケース平均とlog事前値penalty]
    I --> J[自動微分]
    J --> K[勾配clip・Adam更新・parameter clamp]
    K --> L{validation時点か}
    L -->|yes| M[初期観測だけから因果RMSEを計算]
    M --> N{改善したか}
    N -->|yes| O[best stateを保存]
    N -->|no| P[stale countを加算]
    P --> Q{patience到達か}
    Q -->|yes| R[best stateを復元]
    Q -->|no| S{最大epochか}
    O --> S
    L -->|no| S
    S -->|no| D
    S -->|yes| R
    R --> T[test評価・成果物生成]
~~~

| 設定 | 既定値 | 意味 |
|---|---:|---|
| <code>epochs</code> | 80 | 最大epoch |
| <code>batch_size</code> | 12 | 1 batchのケース数 |
| <code>horizon</code> | null | nullは全軌道、数値は区間数 |
| <code>learning_rate</code> | 0.03 | Adam学習率 |
| <code>huber_delta</code> | 1.0 K | Huber切替点 |
| <code>initial_temperature_prior_std</code> | 50.0 K | 隠れ初期modeの事前標準偏差 |
| <code>prior_weight</code> | $10^{-4}$ | log multiplier正則化 |
| <code>gradient_clip</code> | 10.0 | 勾配norm上限。0は無効 |
| <code>validation_every</code> | 5 | validation間隔 |
| <code>patience</code> | 8 | 改善なしvalidation回数 |

optimizerはAdam [R10]。<code>horizon</code> が指定された場合、各epochで各ケースから1つの有効windowをseed付きで選ぶ。window開始行に観測があり、その後のtarget区間にも少なくとも1観測が必要である。

### 6.7 評価指標

| 指標 | 初期状態の決め方 | 用途 |
|---|---|---|
| 条件付きRMSE | 軌道内の観測全体から隠れ初期温度を推定 | パラメータが軌道形状を説明できるか |
| 因果RMSE | 最初の観測行だけから初期化して開ループ計算 | model selection |
| 外部予測RMSE | 観測prefixだけで状態同化しfuture suffixを予測 | 配備条件に近い最終評価 |
| 事前モデルRMSE | 未学習パラメータを同じ条件で評価 | 学習による改善量 |

best epochはvalidationケースの平均因果RMSEで選ぶ。条件付きRMSEは将来観測を初期状態推定に使うため、選択指標には用いない。validationケースが空の場合、実装はtrainケースを代用する。この場合は独立model selectionにならないため、正式な評価ではvalidation partitionを必須とする。

報告するRMSEとMAEは、初期化に使った先頭行を除く。ケース平均値に加えて、センサ別、運転相別、最悪ケースを確認する。ケース平均は長時間ケースの過大な重みを防ぐ一方、全観測点に対するpool値とは意味が異なる。

### 6.8 識別可能性

正値制約と低い学習誤差だけでは、パラメータが一意に決まったことを示さない。代表的な相関を表に示す。

| 相関 | 原因 | 必要な励振 |
|---|---|---|
| 熱容量―全熱流scale | 両者を同倍率で変えると時系列が同じ | 熱容量を独立算定して固定 |
| 熱源gain―時定数 | 立上りの大きさと遅れが相互補償 | step、pulse、cool-down |
| 内部―境界コンダクタンス | 同じ支配時定数を作る | 空間温度差と境界温度を別々に励振 |
| 複数熱源 | node weightが近い | 各熱源の単独励振 |
| 隠れ初期温度―熱係数 | 短い軌道では初期差が残る | 十分な履歴、異なる初期条件 |

実装外の推奨診断として、観測予測のパラメータ感度

$$
J_\theta=\frac{\partial\hat y}{\partial\theta}
$$

と近似情報行列

$$
\mathcal I(\theta)=J_\theta^\mathsf{T}WJ_\theta
$$

のrank、特異値、条件数を確認する。$\mathcal I$ がほぼ特異なら、個別パラメータの物理解釈より、予測量または識別可能なパラメータ組合せを重視する。profile likelihood、bootstrap、異なるseed・splitでの再学習も有効であるが、現行CLIはこれらを自動生成しない。

### 6.9 残差診断

データサイエンス上、単一のRMSEだけで採否を決めない。少なくとも次を確認する。

| 診断 | 確認する問題 |
|---|---|
| 残差対時刻・運転相 | 立上り、定常、cool-downの系統差 |
| 残差対予測温度 | 温度依存物性、放射の不足 |
| 残差対操作量・変化率 | 未表現の入力非線形、dead time |
| センサ間残差相関 | ノード分割不足、common-modeずれ |
| 残差自己相関 | 未再現時定数、雑音独立仮定の破れ |
| ケース別worst | 平均値に隠れた運転条件 |
| 事前モデルとの差 | 学習の実質的な増分価値 |

残差に構造が残る場合、epoch追加より先に、時刻規約、入力単位、ノード構造、境界条件を点検する。

## 7. 将来温度予測

### 7.1 予測要求の境界

予測入力は、実測温度を持つ履歴prefixと、温度列がすべて欠測である将来suffixを1本の軌道として表す。観測maskを $M_{k,j}\in\{0,1\}$ とすると、最初の全センサ欠測行

$$
k_f=\min\left\{k\mid \sum_j M_{k,j}=0\right\}
$$

を将来suffixの開始とし、予測起点を

$$
k_o=k_f-1
$$

とする。$k_f$ 以降に観測が再出現する入力は、因果性が曖昧になるため受理しない。初期行には少なくとも1センサの観測が必要で、将来行も1行以上必要である。

履歴中の一部センサ欠測は許容する。したがってprefixとは「各行に少なくとも1観測がある連続区間」であり、全センサが常時そろうことを意味しない。

### 7.2 履歴同化と開ループ計算

予測処理は次の2段階からなる。

1. $k=0$ で物理的初期値を構成し、初回観測を一度だけKalman更新へ反映する。
2. $k_o$ まで観測を時系列順に同化し、その後は将来commandだけを用いて開ループ計算する。

履歴端で推定された未知発熱平均とsensor bias平均は、将来も持続するという根拠がないためゼロに戻す。一方、その不確かさを失わないよう共分散は保持する。温度状態、実効アクチュエータ、および状態間の相関も保持する。

~~~mermaid
sequenceDiagram
    participant CSV as 予測要求CSV
    participant OBS as 因果observer
    participant RC as RC状態方程式
    participant OUT as 予測成果物
    CSV->>OBS: 初期観測・initial_effective
    OBS->>OBS: 初回posterior更新
    loop 履歴区間 0...ko
        CSV->>OBS: command・次時刻観測・mask
        OBS->>RC: 状態予測
        RC-->>OBS: prior
        OBS->>OBS: 観測更新
    end
    OBS->>OBS: 未知発熱平均・bias平均を0へ
    loop 将来区間 ko...K
        CSV->>RC: command・dt
        RC-->>OBS: 予測状態・共分散
        OBS-->>OUT: 温度・標準偏差
    end
~~~

forecast用observerの既定値を表に示す。forecastで上書き可能なのは、表中「可」とした4項目だけである。

| 設定 | 既定値 | forecast上書き | 物理的意味 |
|---|---:|---|---|
| <code>disturbance_process_std</code> | 0.02 W s$^{-1/2}$ | 可 | 未知熱流random walkの強さ |
| <code>bias_process_std</code> | 0 | 不可 | 将来予測ではbias状態を固定 |
| <code>sensor_std</code> | 0.15 K | 可 | 履歴同化時の観測雑音 |
| <code>innovation_gate_sigma</code> | 4.0 | 可 | 外れ値に対する雑音膨張開始点 |
| <code>initial_temperature_std</code> | 100 K | 可 | 隠れ温度の弱事前分布 |
| <code>initial_disturbance_std</code> | 0 | 不可 | 初期未知発熱の分散 |
| <code>initial_bias_std</code> | 0 | 不可 | 初期biasの分散 |
| <code>bias_reference</code> | null | 不可 | forecastではbias推定を使用しない |

<code>initial_effective_&lt;control&gt;</code> がない場合は先頭commandを実効アクチュエータ初期値とする。遅れの大きい装置ではこの仮定が予測初期の主要誤差となり得るため、運転ログから既知ならCSVへ明記する。

### 7.3 予測不確かさ

物理温度状態の共分散を $P_{T,k}$、観測行列を $H_T$ とすると、ノードおよびセンサ温度の標準偏差は

$$
\sigma_{T_i,k}=\sqrt{(P_{T,k})_{ii}},
\qquad
\sigma_{y_j,k}=\sqrt{(H_TP_{T,k}H_T^\mathsf{T})_{jj}}
$$

である。出力する95%区間はGaussian近似により

$$
\hat y_{j,k}\pm1.95996398454\,\sigma_{y_j,k}
$$

とする。

この区間が表すのは、初期状態とobserverの連続時間process noiseから伝播した潜在物理状態の不確かさである。センサが将来返す測定値ではなく、モデルが推定する物理温度の区間であるため、観測雑音分散は加えない。

予測に必要な総不確かさは、概念的には

$$
\operatorname{Var}_{\mathrm{total}}
\approx
\operatorname{Var}_{\mathrm{state/process}}
+
\operatorname{Var}_{\mathrm{parameter}}
+
\operatorname{Var}_{\mathrm{future\ input}}
+
\operatorname{Var}_{\mathrm{model\ form}}
+
\operatorname{Var}_{\mathrm{measurement}}
$$

に分けられる。現行実装が区間へ含めるのは第1項だけである。

| 不確かさ源 | 現行95%区間 | 別途必要な評価 |
|---|---|---|
| 初期温度・未知熱process | 含む | process noise調整 |
| 学習パラメータ | 含まない | bootstrap、profile likelihood、posterior sampling |
| 将来command・境界条件 | 含まない | シナリオ計算、入力誤差伝播 |
| ノード化・未表現非線形 | 含まない | CAE・実験残差、model discrepancy |
| 将来の測定雑音 | 含まない | 測定値予測が必要な場合のみ $R$ を加算 |

保持データでは、名目95%区間の経験的coverage、平均区間幅、予測時間別coverageを確認する。coverageが95%に近いことだけでは十分でなく、過大な区間幅で達成していないかも併記する。

### 7.4 学習範囲との照合

artifactに保存したtrain範囲と予測要求を軸別に比較する。

| 照合量 | 要求側の値 | 学習側の記録 |
|---|---|---|
| control | 起点以降の各commandの最小・最大 | train command範囲 |
| predicted temperature | 予測センサ温度の最小・最大 | train観測温度範囲 |
| time step | 起点以降の各 $\Delta t$ | train $\Delta t$ 範囲 |
| forecast horizon | $t_K-t_{k_o}$ | train軌道継続時間範囲 |
| control slew | 起点直前を含む $\lvert\Delta u/\Delta t\rvert$ | train変化率範囲 |

判定は軸ごとの長方形範囲比較であり、入力間相関、未観測の組合せ、局所的なデータ密度を評価しない。「範囲内」は内挿の保証ではなく、「範囲外」は計算停止でもない。運用側は <code>outside</code> を警告または承認フローへ接続する。

より厳密な外挿検知が必要な場合は、Mahalanobis距離、密度推定、recipe単位の最近傍距離を追加し、trainだけで閾値を設定して保持データで偽警報率を確認する。

### 7.5 予測評価

予測評価用CSVに真値列を置く場合も、実行時の温度入力列は起点以降を欠測とする。真値は <code>truth_*</code> など別namespaceに保持し、予測器から参照させない。

センサ $j$ の未観測suffix $\mathcal F$ に対する指標は

$$
\operatorname{RMSE}_j
=
\sqrt{\frac{1}{\lvert\mathcal F_j\rvert}
\sum_{k\in\mathcal F_j}(\hat y_{j,k}-y_{j,k})^2},
\qquad
\operatorname{MAE}_j
=
\frac{1}{\lvert\mathcal F_j\rvert}
\sum_{k\in\mathcal F_j}\lvert\hat y_{j,k}-y_{j,k}\rvert
$$

とする。平均値だけでなく、ケース別最大誤差、sensor別、運転相別、予測時間別を報告する。学習前モデルまたは単純baselineと同一ケースで比較し、学習済みモデルの増分価値を確認する。

## 8. 因果状態監視

### 8.1 拡張状態

monitorは、物理温度 $\boldsymbol T$、既知source経路に沿う未知熱流 $\boldsymbol d$、識別可能なsensor bias座標 $\boldsymbol\beta$ を同時推定する。

$$
\boldsymbol z=
\begin{bmatrix}
\boldsymbol T\\
\boldsymbol d\\
\boldsymbol\beta
\end{bmatrix}
$$

sourceが1つ以上定義されている場合、未知熱流の空間basis $W_d$ はsource weight行列とする。sourceがない場合はnode単位のidentity basisとする。したがって未知熱流の物理ノード表現は

$$
\boldsymbol q_{\mathrm{unknown}}=W_d^\mathsf{T}\boldsymbol d
$$

である。source weightが互いにほぼ線形従属なら、合成熱流は推定できてもsource別の分離は不安定になる。

連続時間の局所線形系は

$$
\dot{\boldsymbol z}
=
F\boldsymbol z+\boldsymbol g,
\qquad
F=
\begin{bmatrix}
A & C^{-1}W_d^\mathsf{T} & 0\\
0 & 0 & 0\\
0 & 0 & 0
\end{bmatrix}
$$

と書ける。$\boldsymbol g$ は既知source、境界、アクチュエータを含むnominalモデル側で計算する。観測モデルは

$$
\boldsymbol y
=
\underbrace{\begin{bmatrix}H_T&0&B_\beta\end{bmatrix}}_{H_z}
\boldsymbol z+\boldsymbol v,
\qquad
\boldsymbol v\sim\mathcal N(0,R)
$$

である。

### 8.2 biasの基準

絶対温度のcommon-modeずれは、全sensor biasと隠れ温度の移動で説明できるため、biasには基準が必要である。実装は次のいずれかを使う。

| 設定 | 制約 | 解釈 |
|---|---|---|
| <code>bias_reference: null</code> | $\boldsymbol 1^\mathsf{T}\boldsymbol b=0$ | sensor間の相対biasのみ |
| <code>bias_reference: sensor_name</code> | $b_{\mathrm{ref}}=0$ | 校正済みsensorに対する相対bias |
| sensorが1個 | bias状態なし | biasと物理温度を分離不能 |

$\boldsymbol b=B_\beta\boldsymbol\beta$ とする。zero-meanの場合、$B_\beta$ は $\boldsymbol1$ に直交する正規直交basisである。reference方式では、基準sensor行を0、他行をidentityとする。reference sensorはmonitor軌道内に少なくとも1観測を持たなければならない。

### 8.3 process noiseの離散化

未知熱流とbiasを連続時間random walkとし、spectral densityを

$$
Q_c=
\operatorname{diag}
\left(
\boldsymbol0_{n_T},
\sigma_d^2\boldsymbol1_{n_d},
\sigma_b^2\boldsymbol1_{n_\beta}
\right)
$$

とする。exact積分器ではVan Loan法 [R5] により

$$
\exp
\left(
\begin{bmatrix}
F&Q_c\\
0&-F^\mathsf{T}
\end{bmatrix}
\Delta t
\right)
=
\begin{bmatrix}
\Phi&\Gamma\\
0&\Phi^{-\mathsf{T}}
\end{bmatrix},
\qquad
Q_d=\Gamma\Phi^\mathsf{T}
$$

を計算する。予測共分散は

$$
P_k^-=\Phi_kP_{k-1}^+\Phi_k^\mathsf{T}+Q_{d,k}
$$

である。implicit積分器では、状態遷移

$$
\Phi=(I-\Delta tF)^{-1}
$$

と整合する近似

$$
Q_d=\Phi(Q_c\Delta t)\Phi^\mathsf{T}
$$

を用いる。いずれも熱流random walkが熱容量とネットワークを通して温度へ伝わる相関を保持する。

### 8.4 観測更新とrobust化

利用可能なsensor行だけを抜き出した観測行列を $H_{z,k}$ とし、

$$
\boldsymbol\nu_k
=
\boldsymbol y_k-H_{z,k}\hat{\boldsymbol z}_k^-,
\qquad
S_k=H_{z,k}P_k^-H_{z,k}^\mathsf{T}+R
$$

を計算する。外れ値が1センサの更新を過度に支配しないよう、各sensorについて

$$
s_i=
\max\left(
1,\,
\frac{\lvert\nu_i\rvert}
{\gamma\sqrt{(S_k)_{ii}}}
\right),
\qquad
R_{\mathrm{eff}}=
\operatorname{diag}\left((\sigma_ys_i)^2\right)
$$

とする。$\gamma$ は <code>innovation_gate_sigma</code> である。更新gainは

$$
K_k
=
P_k^-H_{z,k}^\mathsf{T}
\left(
H_{z,k}P_k^-H_{z,k}^\mathsf{T}+R_{\mathrm{eff}}
\right)^{-1}
$$

であり、逆行列を構成せず線形方程式として解く。平均と共分散の更新は

$$
\hat{\boldsymbol z}_k^+
=
\hat{\boldsymbol z}_k^-+K_k\boldsymbol\nu_k
$$

$$
P_k^+
=
(I-K_kH_{z,k})P_k^-(I-K_kH_{z,k})^\mathsf{T}
+K_kR_{\mathrm{eff}}K_k^\mathsf{T}
$$

とする。後者はJoseph formで、有限精度下で共分散の対称性と半正定値性を保ちやすい [R4][R14]。

### 8.5 NISと異常判定

出力するnormalized innovation squaredは、robust化前のinnovation covarianceを用いて

$$
\operatorname{NIS}_k
=
\boldsymbol\nu_k^\mathsf{T}S_k^{-1}\boldsymbol\nu_k
$$

とする。自由度は時刻 $k$ で観測できたsensor数

$$
\nu_k^{\mathrm{dof}}=\sum_j M_{k,j}
$$

である。線形Gaussianモデル、正しい $Q/R$、独立なinnovationという条件下では、NISは概ね自由度 $\nu_k^{\mathrm{dof}}$ の $\chi^2$ 分布に従う。

中核実装はNIS、自由度、innovationを出力するが、alarmの閾値・継続条件は決めない。配備時は欠測パターンごとに理論閾値

$$
\chi^2_{\nu_k^{\mathrm{dof}},\,1-\alpha}
$$

を出発点とし、正常保持データで時刻当たり偽警報率、run単位偽警報率、連続超過回数を校正する。自己相関、モデル誤差、多重比較があるため、単一点の理論p値だけで故障判定してはならない。

### 8.6 欠測時の挙動

| 状況 | 平均状態 | 共分散 | 出力 |
|---|---|---|---|
| 初期行 | 利用可能sensorから初期化し、1回更新 | 観測方向を収縮 | prior、innovation、NISはNaN、dof=0 |
| 一部sensor欠測 | 観測行を縮約して更新 | 観測可能部分のみ収縮 | 欠測sensorのinnovationはNaN |
| 全sensor欠測 | priorをposteriorとして保持 | process noise分だけ拡大 | NISはNaN、dof=0 |
| 観測復帰 | その時点のpriorから更新 | 相関を介して全状態へ反映 | 通常のinnovation/NIS |

欠測窓でposterior温度が出力されても、それは測定された値ではない。欠測が長いほど共分散が拡大することを確認し、推定値と実測値を表示上も区別する。

### 8.7 monitor設定

| 設定 | 既定値 | 調整時の観点 |
|---|---:|---|
| <code>disturbance_process_std</code> | 0.02 W s$^{-1/2}$ | 未知熱流追従性とnoise増幅 |
| <code>bias_process_std</code> | 0.005 K s$^{-1/2}$ | drift追従性と物理温度への漏れ |
| <code>sensor_std</code> | 0.15 K | 校正試験の再現性、量子化 |
| <code>innovation_gate_sigma</code> | 4.0 | 外れ値抑制と故障追従性 |
| <code>initial_temperature_std</code> | 1.0 K | 初期温度の信頼度 |
| <code>initial_disturbance_std</code> | 0.5 W | 起動時の未知発熱 |
| <code>initial_bias_std</code> | 0.5 K | 起動時のsensor offset |
| <code>bias_reference</code> | null | zero-meanまたは校正済みsensor |

$\sigma_d$ と $\sigma_b$ は同じ残差を説明し得るため、同時に大きくしない。既知のheat injection、既知のsensor offset、正常運転noiseを別ケースで与え、推定遅れ、定常bias、誤帰属率を比較して調整する。

### 8.8 monitor出力の意味

| 列族 | 意味 | 使用上の注意 |
|---|---|---|
| <code>measured_*</code> | 入力sensor値 | 欠測はNaN |
| <code>prior_physical_*</code> | 観測更新前の物理温度 | sensor biasを含まない |
| <code>predicted_measurement_*</code> | prior物理温度＋prior bias | innovationの基準 |
| <code>posterior_physical_*</code> | 更新後の物理温度 | 測定値そのものではない |
| <code>reconstructed_measurement_*</code> | posterior物理温度＋posterior bias | 観測再構成 |
| <code>state_*</code> | 全ノードposterior温度 | 未観測nodeを含む |
| <code>disturbance_*_w</code> | nodeへ写像した未知熱流 | 正が加熱、負が未表現冷却 |
| <code>sensor_bias_*</code> | 選択したgauge内のsensor offset | 絶対biasとは限らない |
| <code>innovation_*</code> | 測定－prior予測 | 欠測時NaN |
| <code>innovation_std_*</code> | robust化前の予測標準偏差 | gate判定のscale |
| <code>nis</code> / <code>nis_dof</code> | 多変量innovation統計量 / 自由度 | 閾値は運用側で校正 |
| <code>effective_*</code> | アクチュエータ遅れ後の操作量 | raw commandと区別 |

## 9. 全体処理フロー

### 9.1 end-to-endフロー

~~~mermaid
flowchart TD
    A[config.yaml] --> B[設定schema・未知key検査]
    C[system.yaml] --> D[物理schema・単位・参照整合検査]
    E[軌道CSV] --> F[時刻・列・有限値・mask検査]
    B --> G{workflow}
    D --> G
    F --> G

    G -->|train| H[case split]
    H --> I[隠れ初期温度をSVD推定]
    I --> J[RC rollout・Huber損失]
    J --> K[Adam更新]
    K --> L{validation改善}
    L -->|継続| I
    L -->|終了| M[best state復元]
    M --> N[train/val/test評価]
    N --> O[artifact・指標・split保存]

    G -->|forecast| P[artifact hash・schema検査]
    P --> Q[観測prefixを因果同化]
    Q --> R[future suffixを開ループ計算]
    R --> S[95%区間・学習範囲照合]
    S --> T[予測CSV・summary・manifest]

    G -->|monitor| U[artifact hash・schema検査]
    U --> V[拡張状態を初期化]
    V --> W[predict]
    W --> X{観測あり}
    X -->|yes| Y[robust Kalman update・NIS]
    X -->|no| Z[priorを保持]
    Y --> AA{次時刻}
    Z --> AA
    AA -->|あり| W
    AA -->|なし| AB[monitor CSV・summary・manifest]

    O --> AC[staging成功時のみ配置]
    T --> AC
    AB --> AC
~~~

### 9.2 fail-fast順序

| 順序 | 検査 | 失敗時 |
|---:|---|---|
| 1 | config root、未知key、型 | 計算前に例外 |
| 2 | 入出力pathとoverwrite | 既存成果物を変更せず例外 |
| 3 | system schema、名称参照、正値条件 | model生成前に例外 |
| 4 | artifact schema、SHA-256、buffer整合 | 推論前に例外 |
| 5 | CSV列、時刻単調性、shape、有限値 | 当該workflowを中止 |
| 6 | sensor/control順序 | rollout前に例外 |
| 7 | forecast prefix/suffix、bias reference | 状態推定前に例外 |
| 8 | 数値有限性、loss、solve | stagingを破棄して例外 |
| 9 | JSONのNaN非許容 | 完成成果物の置換前に例外 |

複数ケースを処理中に1ケースが失敗した場合も、target directory全体を完成版へ置換しない。部分成果物を正式出力として残さないことを優先する。

### 9.3 責務境界

| 判断 | 中核実装 | 利用側 |
|---|---|---|
| RC構造と単位 | schemaと数値契約を検査 | ノード化の妥当性を承認 |
| パラメータ学習 | 指定範囲・lossで最適化 | 励振設計と識別可能性を確認 |
| 将来温度 | 状態区間を計算 | 安全margin、意思決定閾値を設定 |
| 学習範囲外 | 軸別警告を生成 | 継続・停止・再解析を決定 |
| monitor統計量 | innovation/NISを生成 | alarm、継続条件、対処を規定 |
| CAE比較 | 独立workflowで算出 | mesh・物理仮定・適用可否を承認 |

## 10. 成果物、CLIおよび運用契約

### 10.1 学習成果物

学習run directoryには次を保存する。

| 成果物 | 内容 |
|---|---|
| <code>metrics_by_case.csv</code> | split、条件付き・因果指標をケース別に記録 |
| <code>metrics_by_sensor.csv</code> | split、ケース、sensor別の指標 |
| <code>metrics_summary.json</code> | split別の平均、中央値、worst-case |
| <code>training_history.csv</code> | epoch、loss、validation指標 |
| <code>split.csv</code> | 各caseのtrain/val/test割当 |
| <code>data_summary.csv</code> | 行数、時間範囲、欠測率等の入力要約 |
| <code>config.snapshot.yaml</code> | 実行時configの監査用snapshot |
| <code>test_predictions/</code> | testケースの予測と事前モデル予測 |
| <code>artifact/</code> | 配備可能なmodel、system、metadata |

### 10.2 artifact schema version 4

| ファイル | 内容 | load時検査 |
|---|---|---|
| <code>model.pt</code> | PyTorch <code>state_dict</code> | <code>weights_only=True</code>、shape、buffer |
| <code>system.yaml</code> | schema version 3の物理構造 | schema、名称、正値条件 |
| <code>metadata.json</code> | model type、dtype、integrator、学習・評価、物理パラメータ、hash | schema=4、model type、fitted値整合 |

<code>metadata.json</code> 内の <code>file_sha256</code> は <code>model.pt</code> と <code>system.yaml</code> を対象とする。これは偶発的破損やファイル取り違えを検出するchecksumであり、秘密鍵による署名ではない。攻撃者がmetadataも同時に書き換えられる環境では真正性を保証しない [R15]。

load時は <code>system.yaml</code> からmodelを再構成し、保存stateのderived bufferがsystem定義と一致すること、metadataの物理パラメータがload後modelと数値許容差内で一致することを確認する。

### 10.3 forecast成果物

| 成果物 | 内容 |
|---|---|
| <code>&lt;case_id&gt;.csv</code> | sensor予測、95%区間、node状態、標準偏差、実効control |
| <code>forecast_summary.csv</code> | 履歴行数、起点、出力行数、学習範囲判定 |
| <code>forecast_coverage.csv</code> | quantity/nameごとのtrain範囲とrequest範囲 |
| <code>run_manifest.json</code> | config、artifact、入力hash、解決済みobserver、不確かさscope |

ケースCSVの主要列族は <code>temperature_*</code>、<code>temperature_std_*</code>、<code>temperature_lower_95_*</code>、<code>temperature_upper_95_*</code>、<code>state_*</code>、<code>state_std_*</code>、<code>effective_*</code> である。

### 10.4 monitor成果物

| 成果物 | 内容 |
|---|---|
| <code>&lt;case_id&gt;.csv</code> | §8.8の時刻別推定量 |
| <code>monitor_summary.csv</code> | innovation RMSE、最大絶対innovation、平均NIS/dof |
| <code>run_manifest.json</code> | config、artifact、入力hash、observer、basis |

<code>run_manifest.json</code> はschema version 1で、forecast/monitor入力ファイルとconfigのSHA-256、artifactのmetadataおよびmodel/system hash、package version、解決済み設定を保存する。現行の学習成果物は入力CSVごとのhash、OS、BLAS、GPU driver、Git commitを記録しないため、規制対応または厳密再現が必要なら実行wrapperで補完する。

### 10.5 原子的出力

各workflowはtargetと同じ親directoryに一時staging directoryを作り、全成果物の生成成功後にtargetへ置換する。既存targetを上書きする場合は一時backupへ移し、置換失敗時に復元する。targetがproject root、その祖先、filesystem rootの場合は拒否する。相対output pathはconfig directory配下に限定する。

### 10.6 CLI

標準実行例を示す。

~~~powershell
uv sync --extra dev --locked
uv run --locked --with-editable . celltemp train --config config.yaml
uv run --locked --with-editable . celltemp forecast --config config.yaml
uv run --locked --with-editable . celltemp monitor --config config.yaml
~~~

設定上書きはcommand末尾にdot pathで指定する。

~~~powershell
uv run --locked --with-editable . celltemp forecast --config config.yaml forecast.observer.sensor_std=0.10 forecast.overwrite=true
~~~

未知key、型不一致、forecastで許可されていないobserver項目は受理しない。相対pathの基準は現在directoryではなくconfig fileのdirectoryである。

### 10.7 再現性とセキュリティ

| 項目 | 現行対策 | 残る制約 |
|---|---|---|
| 依存関係 | <code>uv.lock</code> と <code>--locked</code> | OS/accelerator差 |
| 乱数 | top-level <code>seed</code> | backendごとの演算差 |
| config | snapshotおよびruntime hash | 学習元config hashは個別保存しない |
| 入力 | forecast/monitorでfile hash | train入力file hashは未保存 |
| artifact | schema、checksum、buffer整合 | 電子署名なし |
| 出力 | staging後の原子的置換 | filesystem障害時の外部backupなし |
| model load | <code>weights_only=True</code> | 信頼境界外のartifact配布管理は別途必要 |

repository内にLICENSE、COPYING、NOTICEは存在しない。社外配布、組込み、派生物公開の前に、著作権者と利用許諾を明文化する。

## 11. 検証および妥当性評価

### 11.1 用語と証拠の強さ

精度値を解釈する前に、確認対象を区別する。

| 区分 | 問い | 本repositoryでの主な証拠 |
|---|---|---|
| ソフトウェア検証 | 仕様どおり計算するか | unit、property、integration、architecture試験 |
| 数値検証 | 離散化と実装が基準解に一致するか | 厳密解、zero-step、受動性、exact/implicit比較 |
| モデル妥当性評価 | RC縮約が対象現象を再現するか | 保持データ、TopCell、COMSOL比較 |
| CAE解の資格確認 | 参照CAEが十分に収束・校正されているか | mesh、時間刻み、熱収支、実験比較 |
| 用途適格性 | 誤差と不確かさが意思決定に十分か | 装置別受入基準、運転範囲、独立試験 |

低いRC対CAE誤差だけでは、両者が実機温度に正しいことを示さない。参照CAEにメッシュ差や境界条件誤差が残る場合、その誤差より小さいモデル差は解像できない。この原則に従い、以下の数値は証拠源ごとの用途境界とともに示す。

### 11.2 ソフトウェア品質ゲート

文書改訂時点の <code>quality.py fast</code> は、format、lint、型検査、unit/property試験をすべて通過した。各gateの正式な定義は [L5] を正とする。

| gate | 範囲 | 本改訂での扱い |
|---|---|---|
| <code>fast</code> | format、lint、types、unit/property | 再実行、PASS |
| <code>pr</code> | 全試験、architecture、branch coverage | 本改訂では未再実行 |
| TopCell benchmark | データ生成、train、forecast、monitor、評価 | 既存成果物 [E1] を検査 |
| 線形COMSOL workflow | 参照生成、学習、外部評価 | 既存成果物 [E2] を検査 |
| 非線形・高忠実度COMSOL | COMSOL solve、mesh、評価 | 既存成果物 [E3][E4][E5][E6][E7] を検査 |

文書だけの変更であっても、リンク切れ、参照ID、Markdown fence、表構造を点検する。配布候補では <code>pr</code> と対象benchmarkを再実行し、生成日時、入力hash、環境をrelease記録へ残す。

### 11.3 TopCell合成benchmark

TopCellは既知の生成モデルを持つ自己完結型benchmarkである。学習探索から除外したforecast 12ケースとmonitor 5ケースを用い、forecastのうち11ケースをcore精度、1ケースを意図的な非線形model gapとして扱う [L7][L8][E1]。

| 項目 | 結果 |
|---|---:|
| benchmark status | pass |
| 判定check | 14 / 14 true |
| core forecastケース | 11 |
| core平均RMSE | 0.1429118874 K |
| core最大RMSE | 1.3026765542 K |
| 未学習事前モデル平均RMSE | 7.1850140685 K |
| 意図的model gap RMSE | 9.8234057479 K |
| monitorケース | 5 |
| 最大検知遅れ | 1.0 s |

この結果は、既知真値に対する回帰防止、外挿警告、履歴初期化、欠測、sensor fault、未知発熱の一連の実装確認として有効である。生成則とRCモデルが近い合成問題であり、実装のpassを実機精度へ外挿しない。

### 11.4 線形COMSOL評価

線形評価はCOMSOL Application LibraryのElectronic Chip Coolingを基に、16学習、8外部forecast、5monitorの計29軌道、9,029行で構成する [L9][L10][E9][R16]。RCのsensor truthは対応領域の体積平均温度であり、局所hotspotとは異なる。予測値は [E2]、監視値は [E8] を正とする。

| 評価 | 結果 | 判定 |
|---|---:|---|
| dataset/workflow check | 10 / 10 true | pass |
| 内部分割test平均RMSE | 0.0269384804 K | 参考 |
| 外部forecast平均RMSE | 0.0362763567 K | 0.25 K screening limit未満 |
| 外部forecast最大RMSE | 0.0829736997 K | F05短pulse |
| 事前モデル平均RMSE | 7.3617894541 K | 比較baseline |
| 平均RMSE改善率 | 99.0555% | 同一外部ケース比較 |
| 最大平均―hotspot差 | 0.23802152 K | 空間縮約差 |
| 最大hotspot過小予測 | 0.32799128 K | 安全側marginに未包含 |

monitorの代表結果を示す。

| ケース | 事象 | 結果 |
|---|---|---:|
| M03 | chip sensorへ+3 K offset | 検知遅れ0 s、最大NIS 403.305 |
| M04 | sensor欠測窓 | 欠測区間posterior RMSE 0.04033 K、有限性維持 |
| M05 | 非公開3 W chip発熱 | 検知遅れ10 s、推定peak 3.182 W、event中最大bias 0.0611 K |

線形CAEは実装と線形縮約の整合性を確認する強い証拠である。一方、体積平均とhotspotの差、材料・境界の不確かさ、実験未比較は別の誤差源である。局所最高温度の判定には使用しない。

### 11.5 非線形COMSOL評価

中忠実度非線形datasetは、3D laminar conjugate heat transfer、温度依存air、任意のsurface-to-surface radiationを含む。COMSOL 6.4.0.429で2026-08-30に全27ケース、2,967行を再計算している [L11][E3][R17][R18]。

| role | ケース | 主目的 |
|---|---:|---|
| train | 10 | power、inlet温度、流速、初期状態の同定 |
| forecast | 9 | 内挿、外挿、短pulse、hot start |
| monitor | 3 | 正常、3 W未知発熱、未知の冷却低下 |
| model gap | 5 | 放射あり/なしのpaired比較 |

| QA項目 | 結果 |
|---|---:|
| 重複 <code>(case_id,time)</code> | 0 |
| truth欠損 | 0 |
| 温度範囲 | 15.00–142.99 °C |
| 熱収支残差絶対99%点 | 0.0240 W |
| 熱収支残差絶対最大 | 0.1096 W |
| 同定validation因果RMSE [E10] | 0.312687 K |
| 同定test因果RMSE [E10] | 0.404787 K |

paired放射ケースでは、基準ケースに対するchip終端温度差が−4.05 Kから−15.15 Kに達し、放射を省略したモデルの適用限界が明瞭に現れる。代表meshは収束条件を満たしていないため、このdatasetはmodel-form screening、workflow評価、試験設計に限定する。絶対温度、hotspot limit、圧力損失の設計truthには使用しない [E3]。

### 11.6 高忠実度COMSOL評価

高忠実度評価は、局所meshを用いたconjugate-flowケースHV01と、その同一入力に放射を加えたHV02のpaired transientである [L12][E4][E5]。公開3入力は学習範囲内で、forecast truthは入力へ渡していない。

| 項目 | 結果 |
|---|---:|
| workflow status | pass |
| HV01 RMSE | 0.0476529870 K |
| HV02 RMSE | 0.1435842440 K |
| 2ケース平均RMSE | 0.0956186155 K |
| 事前モデル平均RMSE | 0.8889186807 K |
| 全scalar誤差が隣接mesh差以内 | true |
| 最大隣接mesh差 | 0.4962085957 K |
| model adequacy | <code>not_resolved_beyond_mesh_difference</code> |

「全誤差がmesh差以内」は、RCモデルが厳密に妥当化されたことを意味しない。参照解の差がRC誤差より大きく、モデル精度をそれ以上に分解できない状態である。

| 資格項目 | 状態 | 結論 |
|---|---|---|
| local-medium benchmark基準 | qualified | 0.5 K screening proxyには使用可 |
| strict mesh基準 | 未達 | mesh独立な設計値ではない |
| transient時間刻み | 未評価 | 時間離散化誤差は未確定 |
| 実験比較 | 未実施 | 実機妥当性は未確認 |
| strict validation status | <code>not_qualified</code> | 用途はmodel-form screeningのみ |

放射pairの真値温度効果は最大0.39275677 Kで、保守的な隣接mesh差に対する比は最大0.7979である。方向は一致したが、paired transientのmesh refinementがないため、放射効果の大きさは定量解像できない。したがって結論はdirectional screeningに留める。

### 11.7 証拠から導ける結論

| 主張 | 支持 | 支持しない主張 |
|---|---|---|
| 共通RC engineが同定・予測・監視を一貫実行する | ソフトウェア試験、TopCell | 任意装置での精度保証 |
| 線形化CAEの平均温度を低誤差で再現した | 線形COMSOL外部8ケース | hotspot安全保証 |
| 流速依存を含む非線形条件でも有限で因果的に動作する | 非線形・高忠実度workflow | mesh独立な絶対温度 |
| 放射省略の方向とmodel gapを検出できる | paired CAE | 放射効果量の厳密同定 |
| sensor offsetと未知熱流を分けて推定できる | 合成・線形monitorケース | あらゆる故障の一意診断 |
| 現行成果物は追跡可能性を持つ | config/input/artifact hash | 暗号学的真正性、完全再現 |

## 12. リスク管理と受入基準

### 12.1 技術リスク

| ID | failure mode | 主な影響 | 検出方法 | 予防・緩和 |
|---|---|---|---|---|
| RSK-01 | commandと時刻の1行ずれ | 時定数・gainの誤同定 | step応答、区間図との照合 | ZOH規約をデータ生成試験に固定 |
| RSK-02 | W/kW、s/min、K/°C差の混在 | 桁違いの係数と温度 | 値域、エネルギー収支 | SIへ事前変換しschema外で単位管理 |
| RSK-03 | 初期実効actuatorの誤仮定 | 予測初期の系統誤差 | 起動直後の残差 | <code>initial_effective_*</code> を記録 |
| RSK-04 | ノード分割不足 | 自己相関、相別bias | 残差、短pulse、CAE mode | ノード追加または適用範囲縮小 |
| RSK-05 | パラメータ非識別 | seed依存、非物理値 | 感度SVD、profile、再学習 | 固定値、独立励振、事前情報 |
| RSK-06 | recipe漏洩 | 過度に楽観的なtest誤差 | recipe/group監査 | case族単位の明示分割 |
| RSK-07 | 外挿 | 温度誤差増大、符号誤り | coverage表、残差 | 範囲外承認、再学習、CAE |
| RSK-08 | 放射・温度依存物性の省略 | 高温域の系統誤差 | 残差対温度、paired CAE | response law拡張、model discrepancy |
| RSK-09 | 平均温度をhotspotと解釈 | 安全margin不足 | CAEの平均―最大差 | hotspot補正または別モデル |
| RSK-10 | 未知熱とbiasの誤帰属 | 誤診断 | injection/offset分離試験 | $Q/R$、bias gauge、sensor配置を調整 |
| RSK-11 | common-mode bias | 絶対温度ずれを見逃す | 校正基準、冗長sensor | reference sensorまたは独立計測 |
| RSK-12 | 狭すぎる予測区間 | 過信 | 時間別coverage | parameter/input/model-form項を追加 |
| RSK-13 | artifact改ざん | 誤modelの配備 | 署名、配布監査 | checksumに加え署名・access control |
| RSK-14 | 未収束CAEをtruth化 | 誤った受入判断 | mesh/time/experiment qualification | 証拠階層と用途を明記 |

### 12.2 repository release gate

release候補は少なくとも次を満たす。

1. lock済み環境で <code>quality.py pr</code> がpassする。
2. 中核API変更に対応するunit、property、integration試験がある。
3. artifact、system、runtime manifestのschema互換性を確認する。
4. TopCellを再生成から評価まで実行し、全checkがtrueである。
5. 物理式または入出力変更時は、該当するCOMSOLまたは解析解benchmarkを再実行する。
6. benchmark成果物の入力hash、package version、生成日時を保存する。
7. README、単位規約、本仕様書の契約が実装と一致する。
8. licenseと配布条件がrelease範囲に対して明確である。

### 12.3 装置別の採用gate

repositoryのpassは、対象装置への採用承認ではない。装置別に次を定量化する。

| gate | 必要な証拠 | 合否値の決定者 |
|---|---|---|
| 物理範囲 | 材料、接触、境界、入力、初期状態の範囲 | 熱設計責任者 |
| データ品質 | 欠測、同期、校正、recipe独立性 | データ責任者 |
| 識別可能性 | 感度rank、再学習分散、係数妥当性 | 物理・DS共同 |
| 予測精度 | 独立caseのRMSE、最大誤差、phase別誤差 | 製品要求責任者 |
| 区間校正 | 名目coverage、区間幅、horizon依存 | DS・品質保証 |
| hotspot margin | 平均―最大差と不確かさ | 熱設計・安全 |
| monitor性能 | 偽警報率、検出率、遅れ、誤帰属 | 運用・保全 |
| CAE資格 | mesh、時間刻み、熱収支、実験比較 | CAE責任者 |
| 安全構成 | 独立保護、故障時動作、手動復帰 | システム安全責任者 |

閾値は対象装置の許容温度、制御周期、保全費用に依存するため、本パッケージのbenchmark値をそのまま採用しない。合否値、承認者、根拠データ、適用rangeをmodel cardまたはrelease recordへ固定する。

### 12.4 変更影響

拡張時の依存方向と試験方針は [L6] に従う。変更種別ごとの最低限の再評価範囲を次に示す。

| 変更 | 必須の再評価 |
|---|---|
| node、edge、sensor配置 | 観測可能性、識別可能性、全benchmark |
| 熱容量 | 時定数、初期状態、全train |
| response law | exact適用条件、gradient、外挿 |
| integrator | 解析解、zero-step、observer共分散 |
| CSV時刻規約 | 全workflow、境界step試験 |
| lossまたはsplit | model selection、漏洩、指標比較 |
| observer $Q/R$ | coverage、偽警報、検知遅れ、誤帰属 |
| artifact schema | backward compatibility、hash、load安全性 |
| COMSOL mesh/physics | dataset QA、truth差、受入結論 |

## 13. 要求トレーサビリティ

### 13.1 機能要求

| ID | 要求 | 実装 | 主な検証 |
|---|---|---|---|
| FR-01 | SI単位の熱networkを定義できる | [C1][C2][C3] | topology/system unit試験 |
| FR-02 | 正の容量・熱流係数を維持する | [C3] | property、passivity試験 |
| FR-03 | variable $\Delta t$ を直接積分する | [C4] | 解析解、variable-dt試験 |
| FR-04 | 遅れを含む実効actuatorを状態として保持する | [C3][C4] | actuator step、initial effective試験 |
| FR-05 | 軌道単位で漏洩なく分割する | [C6] | split再現性・明示分割試験 |
| FR-06 | 隠れ初期温度を正則化付きで推定する | [C7] | full-rank/rank-deficient試験 |
| FR-07 | 因果RMSEでmodel selectionする | [C8] | trainerおよびworkflow試験 |
| FR-08 | 観測prefixから開ループforecastする | [C5][C9] | prefix境界・未来漏洩試験 |
| FR-09 | 潜在状態の標準偏差と95%区間を出力する | [C5][C9] | covariance・workflow試験 |
| FR-10 | 学習range外を軸別に報告する | [C9] | control/temp/time/slew試験 |
| FR-11 | 未知熱流と相対sensor biasを因果推定する | [C5][C10] | fault/heat/bias gauge試験 |
| FR-12 | 欠測sensorをmaskして更新する | [C5][C10] | sparse/missing試験 |
| FR-13 | NISと自由度を出力する | [C5][C10] | innovation covariance試験 |
| FR-14 | 配備artifactの破損・不整合を拒否する | [C11] | hash/buffer/schema試験 |
| FR-15 | 不完全な出力directoryを正式配置しない | [C12] | staging/rollback試験 |
| FR-16 | forecast/monitorの入力来歴を保存する | [C12] | manifest hash試験 |

### 13.2 非機能要求

| ID | 要求 | 実現方法 | 制約 |
|---|---|---|---|
| NFR-01 | 決定性 | seed、lock file、明示split | hardware差は残る |
| NFR-02 | 数値安定性 | matrix exponential、linear solve、implicit fallback | 大規模networkは計算量増 |
| NFR-03 | 監査性 | snapshot、metrics、manifest、hash | train入力hashは未実装 |
| NFR-04 | 保守性 | domain→engine→workflowの一方向依存 | schema変更は移行設計が必要 |
| NFR-05 | 安全な失敗 | strict validation、fail-fast、staging | 業務alarm処理は利用側責務 |
| NFR-06 | 可搬性 | CPU既定、相対path、自己完結artifact | OS/driver完全同一性は未保証 |
| NFR-07 | 説明性 | 物理係数、熱流、bias、innovationを出力 | 非識別時の係数解釈に制限 |

### 13.3 仕様―証拠対応

| 仕様領域 | 実装証拠 | データ証拠 | 未解決事項 |
|---|---|---|---|
| 熱収支・受動性 | [C3][C4] | TopCell、線形COMSOL | 実機境界の同定 |
| 同定・漏洩防止 | [C6][C7][C8] | split、保持case | parameter uncertainty |
| forecast因果性 | [C5][C9] | [E1][E2][E4] | 長期horizon校正 |
| monitor分離 | [C5][C10] | TopCell/COMSOL故障case | 実機偽警報率 |
| artifact・来歴 | [C11][C12] | manifest/hash試験 | 署名、train入力hash |
| 高忠実度妥当性 | external workflow | [E3][E4][E5][E6][E7] | strict mesh、時間刻み、実験 |

## 14. 参考資料

### 14.1 repository内文書

[L1]: ../README.md "thermal-cell-practical README"
[L2]: units_and_conventions.md "Units and conventions"
[L3]: product_architecture.md "Product architecture"
[L4]: model_notes.md "Model notes"
[L5]: quality.md "Quality gates"
[L6]: developer_extension_guide.md "Developer extension guide"
[L7]: ../benchmarks/topcell/README.md "TopCell benchmark README"
[L8]: ../benchmarks/topcell/docs/topcell_benchmark_problem.md "TopCell benchmark problem"
[L9]: ../external_tools/comsol_chip_cooling/README.md "COMSOL chip cooling workflow"
[L10]: ../external_tools/comsol_chip_cooling/docs/problem_definition.md "Linear COMSOL problem definition"
[L11]: ../external_tools/comsol_chip_cooling/docs/nonlinear_problem_definition.md "Nonlinear COMSOL problem definition"
[L12]: ../external_tools/comsol_chip_cooling/docs/high_fidelity_validation.md "High-fidelity validation definition"

### 14.2 実装

[C1]: ../src/celltemp/domain/topology.py "Physical topology domain model"
[C2]: ../src/celltemp/io/system.py "System YAML input/output"
[C3]: ../src/celltemp/engine/rc.py "Thermal RC engine"
[C4]: ../src/celltemp/engine/integrator.py "Exact and implicit integrators"
[C5]: ../src/celltemp/engine/observer.py "Causal Kalman observer"
[C6]: ../src/celltemp/learning/split.py "Trajectory split"
[C7]: ../src/celltemp/learning/objective.py "Hidden initial state and objective"
[C8]: ../src/celltemp/learning/trainer.py "Training and model selection"
[C9]: ../src/celltemp/workflows/forecast.py "Forecast workflow"
[C10]: ../src/celltemp/workflows/monitor.py "Monitor workflow"
[C11]: ../src/celltemp/artifact.py "Deployment artifact"
[C12]: ../src/celltemp/workflows/common.py "Runtime paths, staging, and manifest"

### 14.3 評価成果物

[E1]: ../benchmarks/topcell/work/outputs/benchmark/benchmark_summary.json "TopCell benchmark summary"
[E2]: ../external_tools/comsol_chip_cooling/reports/cae_benchmark_report/source_data/linear_forecast_metrics.csv "Linear COMSOL forecast metrics"
[E3]: ../external_tools/comsol_chip_cooling/data/nonlinear/quality_report.md "Nonlinear COMSOL data quality report"
[E4]: ../external_tools/comsol_chip_cooling/data/nonlinear_high_fidelity/benchmark/summary.json "High-fidelity benchmark summary"
[E5]: ../external_tools/comsol_chip_cooling/data/nonlinear_high_fidelity/quality_report.md "High-fidelity data quality report"
[E6]: ../external_tools/comsol_chip_cooling/reports/cae_benchmark_report/artifact.json "CAE report provenance"
[E7]: ../external_tools/comsol_chip_cooling/reports/cae_benchmark_report/source_data/headline_summary.csv "CAE case inventory"
[E8]: ../external_tools/comsol_chip_cooling/reports/cae_benchmark_report/source_data/linear_monitor_metrics.csv "Linear COMSOL monitor metrics"
[E9]: ../external_tools/comsol_chip_cooling/data/qa_summary.csv "Linear COMSOL case and row inventory"
[E10]: ../external_tools/comsol_chip_cooling/reports/cae_benchmark_report/source_data/nonlinear_summary.csv "Nonlinear COMSOL identification summary"

### 14.4 外部参考文献

[R1]: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936133 "Y. Zhang, V. Shapiro, and P. Witherell, A Scalable Framework for Process-Aware Thermal Simulation of Additive Manufacturing Processes"
[R2]: https://doi.org/10.1109/TVLSI.2006.876103 "W. Huang et al., HotSpot: A Compact Thermal Modeling Methodology for Early-Stage VLSI Design"
[R3]: https://onlinelibrary.wiley.com/doi/book/10.1002/9781118033029 "L. Farina and S. Rinaldi, Positive Linear Systems: Theory and Applications"
[R4]: https://doi.org/10.1115/1.3662552 "R. E. Kalman, A New Approach to Linear Filtering and Prediction Problems"
[R5]: https://ieeexplore.ieee.org/document/1101743/ "C. F. Van Loan, Computing Integrals Involving the Matrix Exponential"
[R6]: https://epubs.siam.org/doi/10.1137/04061101X "N. J. Higham, The Scaling and Squaring Method for the Matrix Exponential Revisited"
[R7]: https://epubs.siam.org/doi/10.1137/S00361445024180 "C. Moler and C. Van Loan, Nineteen Dubious Ways to Compute the Exponential of a Matrix"
[R8]: https://doi.org/10.1007/BF01963532 "G. G. Dahlquist, A Special Stability Problem for Linear Multistep Methods"
[R9]: https://doi.org/10.1214/aoms/1177703732 "P. J. Huber, Robust Estimation of a Location Parameter"
[R10]: https://arxiv.org/abs/1412.6980 "D. P. Kingma and J. Ba, Adam: A Method for Stochastic Optimization"
[R11]: https://docs.pytorch.org/docs/stable/notes/autograd "PyTorch Autograd mechanics"
[R12]: https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html "PyTorch matrix exponential"
[R13]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html "PyTorch Huber loss"
[R14]: https://onlinelibrary.wiley.com/doi/book/10.1002/0471221279 "Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with Applications to Tracking and Navigation"
[R15]: https://docs.pytorch.org/docs/stable/generated/torch.load.html "PyTorch torch.load"
[R16]: https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_introduction.02.03.html "COMSOL Electronic Chip Cooling tutorial"
[R17]: https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_ug_multiphysics_features.12.12.html "COMSOL Nonisothermal Flow multiphysics coupling"
[R18]: https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/COMSOL_ProgrammingReferenceManual.pdf "COMSOL Programming Reference Manual"
