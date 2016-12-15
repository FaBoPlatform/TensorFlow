# MNISTデータセットのバッチ読み込み

```python
# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

for i in range(5):
    # 20個分のデータ(バッチ)を取得する
    batch = mnist.train.next_batch(20)
    # ラベル(正解データ)と画像データ
    labels, images = batch
    # データ数を表示する
    print len(labels)
```

実行結果

```
Extracting ./MNIST_data/train-images-idx3-ubyte.gz
Extracting ./MNIST_data/train-labels-idx1-ubyte.gz
Extracting ./MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ./MNIST_data/t10k-labels-idx1-ubyte.gz
20
20
20
20
20
```
