# MNISTデータセットの情報

`mnist.train.images`は画像データで784(28の2乗)次元のベクトル。モノクロ画像なので0〜1の濃度情報が格納されている。

`mnist.train.labels`はラベル(正解データ)で1ofKベクトル。数字の7が正解ならば、7番目の要素の値のみが1になっているベクトルとなる。

```python
# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
# 画像データの次元数
print mnist.train.images[0].shape
# ラベルの次元数
print mnist.train.labels[0].shape
```

実行結果

```
Extracting ./MNIST_data/train-images-idx3-ubyte.gz
Extracting ./MNIST_data/train-labels-idx1-ubyte.gz
Extracting ./MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ./MNIST_data/t10k-labels-idx1-ubyte.gz
(784,)
(10,)
```
