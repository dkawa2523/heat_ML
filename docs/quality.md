# Quality gates

## 実行

以下はPowerShell、macOS、Linuxで共通の`uv`コマンドです。

```powershell
uv run --locked python quality.py fast
uv run --locked python quality.py architecture
uv run --locked python quality.py pr
```

`fast`はformat、lint、type、unit/property試験を実行します。`architecture`はRuff、Pyrefly、
import-linter、Radonだけを短時間で実行します。`pr`はこれらに全試験とbranch coverageを加えます。
CIはさらにTopCell product benchmarkと`pip-audit .`を実行し、共有Python環境ではなく、このprojectから
解決されるruntime依存だけを監査します。

## 静的検査の役割

| ツール | 判定するもの | 判定しないもの |
|---|---|---|
| [Ruff](https://docs.astral.sh/ruff/settings/) | format、未使用名、危険な記述、import順、局所的な複雑度 | module責務、物理妥当性 |
| [Pyrefly](https://pyrefly.org/en/docs/configuration/) | product APIと外部adapter境界の型、設定値のnarrowing、戻り値の整合 | CAEデータの品質、数値精度 |
| [import-linter](https://import-linter.readthedocs.io/en/stable/contract_types/) | package間の依存方向、leafの純粋性、workflow間の独立性 | 実行時の正しさ、関数内部の複雑度 |
| [Radon](https://radon.readthedocs.io/en/master/commandline.html) | cyclomatic complexityとmaintainabilityの高リスク箇所 | 閾値以下のコードが読みやすいという保証 |

同じ問題を複数ツールで契約化しない。Pyreflyは`src/celltemp`、`tests`、`benchmarks`、
`external_tools`、`quality.py`を一つの型検査範囲とする。外部scriptはpackage importで実行し、pandasから
合否値を取り出す境界では数値型へ明示変換する。型検査のためだけのprotocolや抽象層は追加しない。

```powershell
uv run --locked python -m pyrefly check
uv run --locked python -m radon cc src/celltemp external_tools benchmarks/topcell -s -a
uv run --locked python -m radon mi src/celltemp external_tools benchmarks/topcell -s
```

## 数値検証

| 対象 | 主な試験 |
|---|---|
| command区間のleft/right規約 | `test_domain.py`, `test_io.py` |
| 可変`dt`とaffine解析解 | `test_engine_model.py`, `test_engine_rollout.py`, `test_inference_forecast.py` |
| 対称熱流・energy conservation | `test_engine_model.py`, `test_physical_properties.py` |
| 受動系の上下限 | `test_engine_model.py`, `test_physical_properties.py` |
| actuator解析解・overshootなし | `test_engine_model.py`, `test_engine_rollout.py`, `test_physical_properties.py` |
| 欠測・隠れnode・履歴posterior handoff | `test_engine_observer.py`, `test_inference_forecast.py` |
| zero-mean / reference sensor bias gauge | `test_inference_monitor.py`, `test_workflows.py` |
| forecast境界後の観測拒否 | `test_inference_forecast.py`, `test_workflows.py` |
| gradientとrollout学習 | `test_learning.py` |
| artifact round-trip | `test_artifact.py` |
| 応答・均一性・操作量指標 | `test_analysis.py` |
| power-stepのZth/Rth適格判定 | `test_thermal_impedance.py`, `test_workflows.py` |
| 熱経路G/R/C・連続時間pole | `test_model_analysis.py`, `test_workflows.py` |
| analyze → train → forecast → monitor | `test_workflows.py` |
| trajectory split leakage | `test_dataset.py` |

## Architecture

`.importlinter`は一方向依存と公開workflow間の独立性を検証します。

```text
cli → workflows → analysis/artifact/learning/inference → engine/io → domain/config
```

domainとconfigは独立したleafです。domainはtorch/pandas/YAMLに依存せず、compute層は
pandas/YAMLを読みません。新moduleはexhaustive layersへ追加しない限りarchitecture検査を
通りません。

Radonの自動gateはproduct coreのD–F complexityまたはC maintainabilityだけを失敗にします。C complexityは
設計レビュー対象として記録するが、条件分岐を小関数へ機械的に移すだけの変更は要求しません。外部scriptは
同じ監査で可視化し、用途を説明できない大きな派生物生成器は分割のために残さず削除します。

## Coverage

全体branch coverage下限は`pyproject.toml`にあります。単なる行実行率を上げる試験ではなく、
解析解、不変量、round-trip、失敗入力を優先します。
