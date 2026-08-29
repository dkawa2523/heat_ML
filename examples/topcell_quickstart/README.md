# TopCell quickstart

このディレクトリだけで、同じ熱RC artifactを使う学習・open-loop予測・実測monitorを試せます。
CSVはすべて自己完結形式で、別のcase一覧やschedule manifestはありません。

```powershell
py -3.13 -m celltemp.cli train --config examples/topcell_quickstart/config.yaml
py -3.13 -m celltemp.cli forecast --config examples/topcell_quickstart/config.yaml
py -3.13 -m celltemp.cli monitor --config examples/topcell_quickstart/config.yaml
```

`config.yaml`からの相対パスで`system.yaml`と`data/`を参照します。生成物は`work/`だけへ保存され、
再実行時は処理が完了した結果だけが置換されます。
