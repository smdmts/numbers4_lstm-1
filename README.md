# LSTMを用いたナンバーズ4の当選番号予測
ニューラルネットワークを用いてナンバーズ4の次回の当選番号を予測する
## ナンバーズ4とは
 - 数字選択式宝くじ
 - 平日は毎日抽選が行われる
 - 数字が4つ並ぶナンバーズ4を解析対象
 
 - 同じ数字が何回か連続して出現しやすい
  → 引っ張り現象 = 前回の結果と今回の結果には相関関係がある
## 実行環境
 - Python 3.5.2 (Anaconda)
## 必要なライブラリ
 - Chainer 3.0.0
 - numpy
## 使い方
### 学習
```train.py```には以下の引数が設定されている。-tでtypeの設定を行うことができる。n4は4桁の数字として学習させ、n4_oneは4桁の数字を分割して一文字ずつ学習させる。
```Python:train.py
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', type=int, default=50,
                    help='Number of examples in each mini-batch')
parser.add_argument('--bproplen', '-l', type=int, default=50,
                    help='Number of words in each mini-batch '
                         '(= length of truncated BPTT)')
parser.add_argument('--epoch', '-e', type=int, default=100,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--gradclip', '-c', type=float, default=5,
                    help='Gradient norm threshold to clip')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--type', '-t', default='n4',
                    help='Choose dataset. n4 or n4_one')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=128,
                    help='Number of LSTM units in each layer')
parser.add_argument('--model', '-m', default='model.npz',
                    help='Model file name to serialize')
args = parser.parse_args()
```
./dataにナンバーズ4の[過去当選番号](http://r7-yosou.hippy.jp/data.html)を配置しておく。また、ファイル形式はcsvを想定している。シート選択等は配慮していないためダウンロード後は、ナンバーズ4のみのファイルに書き換える必要がある。

------------------------------------------------------------------------
### 当選番号の予測
学習後は```predict.py```で予測を行うことができる。-mに学習済みモデル、-pに前回の当選番号を設定し、実行する。
- 実行結果
```terminal
Datasize : 4796
 * train : 1598
 * test  : 1598
 * valid : 1598
 * vocab : 10000

 run train ...

 run test ...

 # ---------------------------------
 # test perplexity : 1.01022369536
 # ---------------------------------
 
 >>> python predict.py -t n4 -m ./model_n4.npz -p 9840 
 9840 6934 9699 6768 ...
```
