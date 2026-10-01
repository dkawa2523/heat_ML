# 日本語問題設定図の編集prompt

生成modeはbuilt-in `image_gen`です。既存の5枚をreference imageとして編集し、装置構造、入熱・冷却位置、
計測領域、case数を維持したまま主要ラベルを日本語化しました。共通仕様は16:9、白背景、科学発表品質です。

## 01_topcell_problem_dataset_method.png

円形4領域、上面プラズマ、下面heater、外周brine、CP / 中央 / 中間 / 端の4計測点を維持する。
右側を「学習 228軌道」「外部予測 12ケース」「監視 5ケース」「4ノード物理RC」「真値／予測」とする。

## 02_linear_comsol_problem_dataset_method.png

chip入熱、一定対流冷却、chip / base / finsの3領域平均を維持する。右側を「学習 16ケース」
「外部予測 8ケース」「監視 5ケース」「3ノード物理RC」「冷却水温度」「真値／予測」とする。

## 03_nonlinear_comsol_problem_dataset_method.png

入口温度・入口風速、共役空気流、chip入熱、表面間放射、3領域平均を維持する。case群を「学習 10ケース」
「放射なし 9ケース」「放射ギャップ 5ケース」とし、右上を「風速依存3ノード物理RC」とする。

## 04_high_fidelity_problem_dataset_method.png

局所細分mesh、同一入力、HV01共役熱流動、HV02放射あり、再学習なしの分岐を維持する。資格状態を
「mesh: screening基準合格／厳格資格なし」「時間刻み: 2 s / 1 s不合格」「実験比較: 未実施」とする。

## 05_common_dataset_evaluation_method.png

1 case = 1 trajectory、case単位の学習／検証・test／外部評価、同定済み物理RC、未学習RC、温度保持、
比較専用ニューラル5モデルを示す。forecast origin以後は将来truthを隠し、known future commandsだけを使う。
下段は物理observerによる未知発熱、sensorずれ、欠測中温度推定を示す。
