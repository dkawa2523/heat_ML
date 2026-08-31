# Quality gates

## 実行

以下はWindows PowerShellの例です。macOS/Linuxでは`py -3`を`python3`へ置き換えます。

```powershell
py -3 quality.py fast
py -3 quality.py pr
```

`fast`はformat、lint、type、unit/property試験を実行します。`pr`は全試験、branch coverage、
architectureを検証します。CIはこれに加えてTopCell product benchmarkと`pip-audit .`を実行し、
共有Python環境ではなく、このprojectから解決されるruntime依存だけを監査します。

## 数値検証

| 対象 | 主な試験 |
|---|---|
| command区間のleft/right規約 | `test_domain.py`, `test_io.py` |
| 可変`dt`とaffine解析解 | `test_engine.py`, `test_inference.py` |
| 対称熱流・energy conservation | `test_engine.py`, `test_physical_properties.py` |
| 受動系の上下限 | `test_engine.py`, `test_physical_properties.py` |
| actuator解析解・overshootなし | `test_engine.py`, `test_physical_properties.py` |
| 欠測・隠れnode・履歴posterior handoff | `test_engine.py`, `test_inference.py` |
| zero-mean / reference sensor bias gauge | `test_inference.py`, `test_workflows.py` |
| forecast境界後の観測拒否 | `test_inference.py`, `test_workflows.py` |
| gradientとrollout学習 | `test_learning.py` |
| artifact round-trip | `test_inference.py` |
| train → forecast → monitor | `test_workflows.py` |
| trajectory split leakage | `test_dataset.py` |

## Architecture

`.importlinter`は一方向依存を検証します。

```text
cli → workflows → artifact/learning/inference → engine/io → domain/config
```

domainとconfigは独立したleafです。domainはtorch/pandas/YAMLに依存せず、compute層は
pandas/YAMLを読みません。新moduleはexhaustive layersへ追加しない限りarchitecture検査を
通りません。

## Coverage

全体branch coverage下限は`pyproject.toml`にあります。単なる行実行率を上げる試験ではなく、
解析解、不変量、round-trip、失敗入力を優先します。
