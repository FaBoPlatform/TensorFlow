# RaspberryPiで学習済みニューラルネットを動かす

RaspberryPi上で学習済みニューラルネットを利用する

## 実行環境

* Raspberry Pi3 Model B
* Raspbian Jessie Lite 2016-09-23
* Python 2.7.9
* Python 3.4.2
* TensorFlow 1.0.0

## サンプルコード

```python
# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# pbファイルのパスを指定する
model_fn = "./mnist_graph.pb"
# pbファイルからグラフ情報を読み込む
f = tf.gfile.FastGFile(model_fn, 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name="")
f.close()

sess = tf.Session()

# オペレーションおよびテンソルの一覧を表示する
for op in sess.graph.get_operations():
    print(op.name, op.outputs)

# テスト用データを1000個利用
acc = sess.run("accuracy:0", feed_dict={"X:0":mnist.test.images[:1000],"t:0":mnist.test.labels[:1000],"keep_prob:0":1.0})

print("accuracy: %f" % acc)
sess.close()
```

MNISTの学習済み畳込みニューラルネットのpbファイルは[こちらのURL](https://www.dropbox.com/s/incxw1qtan68y3a/mnist_graph.pb?dl=0)からダウンロード可能

```
$ curl -O https://www.dropbox.com/s/incxw1qtan68y3a/mnist_graph.pb
```

## 実行結果

```python
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
(u't', [<tf.Tensor 't:0' shape=<unknown> dtype=float32>])
(u'X', [<tf.Tensor 'X:0' shape=<unknown> dtype=float32>])
(u'Reshape/shape', [<tf.Tensor 'Reshape/shape:0' shape=(4,) dtype=int32>])
...略...
(u'add_2', [<tf.Tensor 'add_2:0' shape=(?, 1024) dtype=float32>])
(u'Relu_2', [<tf.Tensor 'Relu_2:0' shape=(?, 1024) dtype=float32>])
(u'keep_prob', [<tf.Tensor 'keep_prob:0' shape=<unknown> dtype=float32>])
...略...
(u'Cast_1', [<tf.Tensor 'Cast_1:0' shape=<unknown> dtype=float32>])
(u'Const_5', [<tf.Tensor 'Const_5:0' shape=(1,) dtype=int32>])
(u'accuracy', [<tf.Tensor 'accuracy:0' shape=<unknown> dtype=float32>])
accuracy: 0.992000
```
