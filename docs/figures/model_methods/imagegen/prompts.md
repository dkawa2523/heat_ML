# 日本語説明図の生成prompt

生成modeはbuilt-in `image_gen`です。すべて新規生成し、共通仕様を16:9、白背景、科学発表向けの
vector-like infographic、navy / teal / orange / purple、高コントラスト、人物なしとしました。
数値結果は画像へ埋め込まず、CSVとmatplotlib図を正本にしています。

## 01_rc_learning_ja.png

題名「物理RCモデルは何を学習するのか」。チップ・ベース・フィンを熱容量`C`のnode、node間を熱伝導`G`、
左を入熱`q`、右を冷却境界として示す。energy balanceを中央に置き、「固定するもの」と「データから同定するもの」
を分離する。観測温度と指令からfull rollout、誤差、自動微分・Adam、係数更新へ戻る循環を描く。
「ニューラルネットではない」「PyTorchは物理係数の最適化に使用」を赤枠で明示する。

## 02_neural_learning_boundary_ja.png

題名「ニューラル時系列モデルに渡す情報と学習」。8 stepのsensor温度、操作量、`dt`履歴と現在区間の操作量を
入力し、`dT/dt`を出力して`T[k+1] = T[k] + dT/dt * dt`で積分する。学習時と評価・予測時を分け、予測時は
将来温度を隠し、予測温度を次の履歴へ戻す。case単位split、因果open-loop検証、外部testを右側に置く。

## 03_neural_architectures_ja.png

題名「5つのニューラル時系列モデル：内部構造の違い」。共通入力・共通出力の間に、MLPのflattenと全結合、
1D-CNNの時間方向局所kernel、TCNのdilation 1/2/4因果畳み込み、GRUの更新gate、LSTMのcell stateと3 gateを
同じ大きさのcardで比較する。各cardは効果が見込める理由と構造上の弱点を短い日本語で示す。

## 04_model_selection_ja.png

題名「熱時系列予測モデルの使い分け」。横軸を学習data量、縦軸を物理解釈から表現柔軟性として、物理RC、MLP、
1D-CNN、TCN、GRU、LSTMを配置する。下段に各modelの強み・弱み、右に同じsplitと予測起点、波形・RMSE・R²・
worst case、複数seed、改善時のみ採用という判定flowを置く。「複雑なモデルを先に選ばない」を強調する。
