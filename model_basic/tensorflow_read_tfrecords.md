# TFRecords形式のファイルを読み込む

バイナリフォーマットである**TFRecords**形式からデータを読み込む。

## サンプルコード

`tf.parse_single_example(serialized, features, name=None, example_names=None)` : 

文字列のTensorから辞書型のデータを取得する

* `serialized` : 文字列のTensor
* `features` : FixedLenFeatureを持つ辞書
* `name` : オペレーション名

扱うTFRecords形式のフォーマット：

|キー|内容|
|:-:|:-:|
|width|画像の幅|
|height|画像の高さ|
|channels|画像のチャンネル数|
|label|画像のラベル|
|image|画像のバイトデータ|

TensorFlowのグラフ : 

![](/img/graph_read_tfrecord.jpg)

```python
#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
from random import randint
import tensorflow as tf

WIDTH = 50
HEIGHT = 50
CHANNELS = 3

filenames = ["./dataset.tfrecords"]
filename_queue = tf.train.string_input_producer(filenames, name="input_producer")
reader = tf.TFRecordReader(name="tfrecord_reader")
key, value = reader.read(filename_queue, name="reader_read")

# 特徴量を読み込む
features = tf.parse_single_example(value, features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'channels': tf.FixedLenFeature([], tf.int64)},
          name="parse_single_example")

image = tf.image.decode_png(features['image'], name="decode_png")  # PNGをデコードする
image.set_shape([WIDTH, HEIGHT, CHANNELS])  # shapeを設定する

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for _ in range(5):
    _features, _image = sess.run([features, image])
    print(_features['label'], _features['width'], _features['height'], _features['channels'], )
    print(_image)

coord.request_stop()
coord.join(threads)
sess.close()
```

## 実行結果

```
0 50 50 3
[[[204]
  [204]
  [204]
  ...,
  [204]
  [204]
  [204]]
(略)
  ...,
  [204]
  [204]
  [204]]]
```

## 実行環境

* Python 3.6.0
* TensorFlow 1.0.0

## 参考

* https://www.tensorflow.org/api_docs/python/tf/TFRecordReader
* https://www.tensorflow.org/api_docs/python/tf/parse_single_example
