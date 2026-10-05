# 保存済みCOMSOLモデルの取り込み

解析済みの3D固体伝熱モデルから、通常の`system.yaml`と自己完結した`trajectory.csv`を作ります。
COMSOLのインストール・対象モデルを読めるlicenseが必要です。モデルをコピーとして読み、保存済み過渡解を
評価します。元の`.mph`は保存せず、新しいsolveも行いません。生成物の利用にはCOMSOLを必要としません。

## 用意するもの

1. **熱回路template**：既存の[system形式](../../examples/topcell_quickstart/system.yaml)で、代表node、
   熱結合、入熱・冷却、sensor対応を記載します。nodeの`heat_capacity`は省略でき、抽出値で補います。
   接続や実効conductance、学習する係数は利用者が決めます。
2. **領域対応**：各nodeにCOMSOLのdomain selection名、または`2,3`のようなdomain番号を対応付けます。
3. **入力対応**：各actuatorに保存解でglobal評価できるCOMSOL式を対応付けます。空間場から熱量を得る場合は
   元モデルの積分couplingなどを使います。単位はtemplateの`actuator.unit`を用い、省略時は式の単位のままです。

templateと抽出温度は同じ尺度で用意します。既定は`degC`、Kなら`--temperature-unit K`を指定し、
冷却温度control・reservoirの定数と係数もKに揃えます。sensor/node名とcontrol名は通常CSV同様に区別します。

## 実行例

repository rootで実行します。次は既存chip_coolingモデルがローカルに保存されている場合の例です。
自分のモデルではmodel/templateのpath、領域、式を置き換えます。

```powershell
uv run --locked --with-editable . python -m external_tools.comsol_import `
  --model external_tools/comsol_chip_cooling/work/models/electronic_chip_cooling_dataset.mph `
  --template external_tools/comsol_chip_cooling/system.yaml `
  --region chip=sel1 --region sink_base=2 --region fins=3,5,6,7 `
  --control 'chip_power=power_cmd(t)' --control 'coolant_temperature=coolant_cmd(t)' `
  --output-dir external_tools/comsol/work/imported
```

過渡Solution datasetが一つなら自動選択し、複数なら`--dataset dset3`のように指定します。
componentが`comp1`以外なら`--component`、熱物性式が異なる場合だけ`--capacity-expression`を指定します。
既存出力を置換する場合だけ`--overwrite`を付けます。失敗時は既存出力を保持します。
CLIの相対pathは実行directory基準です。

## 生成物の利用

出力は`system.yaml`と`trajectory.csv`の2ファイルです。CSVは保存時刻、templateのsensor列、control列を持ち、
そのまま通常のanalyze/trainへ渡せます。次のconfigを`external_tools/comsol/work/config.yaml`へ置けば、この例を解析できます。

```yaml
system: imported/system.yaml
data:
  directory: imported
  pattern: trajectory.csv
  control_convention: right
analysis:
  output_dir: analysis
```

```powershell
uv run --locked --with-editable . celltemp analyze --config external_tools/comsol/work/config.yaml
```

**指令の時刻対応は元モデルへ合わせます。** 抽出は保存時刻の式値を保持します。
行kの指令が次区間に属するなら既定`left`、完了した直前区間に属するなら`right`です。
このchip_coolingモデルのPiecewise関数は完了区間の値を返すため、上の例は`right`を指定しています。
時刻を自動でずらしたり、連続入力から別の指令を推測したりはしません。

熱容量は`∫ρCp dV`の最初の保存値、代表温度は各時刻の`∫ρCp T dV / ∫ρCp dV`です。
sensor値はtemplateの観測行列で代表温度から生成します。物理的な点probe・最高温度の自動抽出ではありません。
温度依存物性を持つ詳細モデルも、生成した熱回路では最初の容量を固定値として扱います。
領域の集約と観測の意味を確認し、必要ならtemplateを直して再生成します。

学習には独立した複数のtrajectoryを用意し、通常のcase単位splitを使います。取り込みの成功は対象装置の
精度保証を意味しません。詳細CAEの空間場・mesh・solver、用途別比較は外部toolの責務です。
複数のouter parameter caseを含む解は、この入口ではcaseごとに分けて取り込みます。
