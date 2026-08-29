# TopCell quickstart

このディレクトリだけで、同じ熱RC artifactを使う学習・open-loop予測・実測monitorを試せます。
CSVはすべて自己完結形式で、別のcase一覧やschedule manifestはありません。

```powershell
celltemp train --config examples/topcell_quickstart/config.yaml
celltemp forecast --config examples/topcell_quickstart/config.yaml
celltemp monitor --config examples/topcell_quickstart/config.yaml
```

`config.yaml`からの相対パスで`system.yaml`と`data/`を参照します。生成物は`work/`だけへ保存され、
再実行時は処理が完了した結果だけが置換されます。
