# 画像を読み込む

画像ファイルから整数値(uint8)のデータを読み込むには次の関数を用いる。

* [tf.image.decode_gif](https://www.tensorflow.org/api_docs/python/tf/image/decode_gif)
* [tf.image.decode_jpeg](https://www.tensorflow.org/api_docs/python/tf/image/decode_jpeg)
* [tf.image.decode_png](https://www.tensorflow.org/api_docs/python/tf/image/decode_png)

## サンプルコード

構築されるTensorFlowグラフ：

![](/img/graph_read_img.jpg)

```python
#!/usr/bin/env python
# coding:utf-8
"""
filename: read_img.py
Read PNG format images with TensorFlow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import tensorflow as tf


filenames = glob('./*.png')  # ['./0.png', './1.png', ... ]
filename_queue = tf.train.string_input_producer(filenames, name="input_producer")
reader = tf.WholeFileReader(name="wholefile_reader")
key, value = reader.read(filename_queue, name="reader_read")
image = tf.image.decode_png(value, name="image")  # jpg->tf.image.decode_jpeg(...)
image_as_float = tf.divide(tf.cast(image, tf.float32), 255., name="image_as_float")

sess = tf.Session()
coord = tf.train.Coordinator()  # スレッドを管理するクラス
# `sess.run(...)`を実行する前に`tf.train.start_queue_runners(...)`を実行する
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for _ in range(10):
    _key, _image = sess.run([key, image])
    print(_key, _image)  # ファイル名, 画像のRGB(整数値)
    _image_as_float = sess.run(image_as_float)
    print(_image_as_float)

# スレッドの停止を要求する
coord.request_stop()
# スレッドの停止を待ち合わせる
coord.join(threads)
sess.close()
```

## 実行結果

ダミー画像をダウンロードする

```
$ for i in {0..10};do curl "https://placehold.jp/50x50.png?text=$i" -o "$i.png"; sleep 1; done;
$ ls *.png
0.png  1.png  10.png 2.png  3.png  4.png  5.png  6.png  7.png  8.png  9.png
```

`python read_img.py` : 

```
b'./9.png' [[[204]
  [204]
  [204]
  ...,
  [204]
  [204]
  [204]]

 [[204]
  [204]
  [204]
  ...,
(略)
 [[ 0.80000001]
  [ 0.80000001]
  [ 0.80000001]
  ...,
  [ 0.80000001]
  [ 0.80000001]
  [ 0.80000001]]]
```

## 参考

* https://www.tensorflow.org/api_docs/python/tf/WholeFileReader
