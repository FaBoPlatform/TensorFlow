# MNISTデータセットの読み込み

学習用データ、評価用データ、テスト用データを読み込む

```python
# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# 訓練用データセット
# 正解データ
# mnist.train.labels
# 画像データ
# mnist.train.images
# 訓練用データセット数
print len(mnist.train.labels)

# 評価用データセット
# mnist.validation.labels
# mnist.validation.images
# 評価用データセット数
print len(mnist.validation.labels)

# テスト用データセット
# mnist.test.labels
# mnist.test.images
# テスト用データセット数
print len(mnist.test.labels)
```

実行結果

```
Extracting ./MNIST_data/train-images-idx3-ubyte.gz
Extracting ./MNIST_data/train-labels-idx1-ubyte.gz
Extracting ./MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ./MNIST_data/t10k-labels-idx1-ubyte.gz
55000
5000
10000
```
