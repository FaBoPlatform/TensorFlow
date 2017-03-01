# TFRecords形式のファイルを書き込む

**TFRecords**はバイナリ形式のシーケンシャルファイルで、大規模データのストリーミングに向いたファイル形式である。

## サンプルコード

PNG形式の画像とそのラベルをTFRecords形式のファイルとして書き出すプログラムを作成する。

扱うTFRecords形式のフォーマット：

|キー|内容|
|:-:|:-:|
|width|画像の幅|
|height|画像の高さ|
|channels|画像のチャンネル数|
|label|画像のラベル|
|image|画像のバイトデータ|

TensorFlowのグラフ : 

![](/img/graph_write_tfrecord.jpg)

```python
#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
from random import randint
import tensorflow as tf

# 画像のサイズ
WIDTH = 50
HEIGHT = 50
CHANNELS = 3

def _bytes_feature(value):
    """バイナリとして書き込む"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """整数値(int64)として書き込む"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

filenames = glob('./*.png')  # ['./0.png', './1.png', ... ]
filename_queue = tf.train.string_input_producer(filenames, name="input_producer")
reader = tf.WholeFileReader(name="wholefile_reader")
key, value = reader.read(filename_queue, name="reader_read")

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

filename = 'dataset.tfrecords'  #  書き込む先のファイル名
print('Writing', filename)
writer = tf.python_io.TFRecordWriter(filename)
for _ in range(10):
    _value = sess.run(value)
    # 書き込むデータのキーと内容を指定する
    example = tf.train.Example(features=tf.train.Features(feature={
                'width': _int64_feature(HEIGHT),
                'height': _int64_feature(WIDTH),
                'channels': _int64_feature(CHANNELS),
                'label': _int64_feature(randint(0, 1)),
                'image': _bytes_feature(_value)}))
    writer.write(example.SerializeToString())  #  書き込む
writer.close()  # 書き込み終了

coord.request_stop()
coord.join(threads)
sess.close()
```

## 実行結果

```
Writing dataset.tfrecords
```

## 実行結果

* Python 3.6.0
* TensorFlow 1.0.0

## 参考

* https://www.tensorflow.org/api_docs/python/tf/TFRecordReader
