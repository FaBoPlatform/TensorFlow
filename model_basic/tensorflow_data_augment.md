# データ拡張

学習を行う前にデータの数を水増しする前処理をデータ拡張(data augument)と呼び、画像データの場合、データ拡張には画像の左右反転やコントラスト変化、明るさ変化等が行われる。

TensorFlowにはデータ拡張を行うための関数が用意されている(以下を参照)。

* https://www.tensorflow.org/api_guides/python/image

## サンプルコード

データ拡張を行うプログラムを作成する。

```python
#!/usr/bin/env python
# coding:utf-8
"""
filename: data_augument.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def random_crop(img):
    """ランダムに画像を切り取る"""
    _proc = tf.random_crop(img, [100, 100, 3])
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)  # 画像のバイナリを返す

def random_flip_left_right(img):
    """ランダムに左右反転する"""
    _proc = tf.image.random_flip_left_right(img)  # 左右反転
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)  # 画像のバイナリを返す

def random_flip_up_down(img):
    """ランダムに上下反転する"""
    _proc = tf.image.random_flip_up_down(img)  # 上下反転
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)  # 画像のバイナリを返す

def random_brightness(img):
    """ランダムに輝度を変更する"""
    _proc = tf.image.random_brightness(img, max_delta=63)
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)

def random_contrast(img):
    """ランダムにコントラストを変更する"""
    _proc = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)

def image_standardization(img):
    """画像を白色化(正規化)する"""
    _proc = tf.image.per_image_standardization(img)
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)

def data_augment(img):
    """上記の前処理を連結する"""
    _proc = tf.random_crop(img, [100, 100, 3])
    _proc = tf.image.random_flip_left_right(_proc)
    _proc = tf.image.random_flip_up_down(_proc)
    _proc = tf.image.random_brightness(_proc, max_delta=63)
    _proc = tf.image.random_contrast(_proc, lower=0.2, upper=1.8)
    _proc = tf.image.per_image_standardization(_proc)
    img_as_int8 = tf.cast(_proc*255.0, tf.uint8)
    return tf.image.encode_png(img_as_int8)

def to_img(contents, name):
    """PNGのバイナリを書き出す"""
    with open(name, 'wb') as f:
        f.write(contents)
        f.close()

filenames = ["./Lenna.png"]
filename_queue = tf.train.string_input_producer(filenames, name="input_producer")
reader = tf.WholeFileReader(name="wholefile_reader")
key, value = reader.read(filename_queue, name="reader_read")
image = tf.image.decode_png(value, name="image")
image_as_float = tf.cast(image, tf.float32, name="image_as_float") / 255.

# データの前処理オペレーション群
random_crop_op = random_crop(image_as_float)
random_flip_left_right_op = random_flip_left_right(image_as_float)
random_flip_up_down_op = random_flip_up_down(image_as_float)
random_brightness_op = random_brightness(image_as_float)
random_contrast_op = random_contrast(image_as_float)
standardization_op = image_standardization(image_as_float)
data_augment_op = data_augment(image_as_float)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# データの前処理を実行する
_crop, _left_right, _up_down, _brightness, _contrast, _std = sess.run([random_crop_op, random_flip_left_right_op, random_flip_up_down_op, random_brightness_op, random_contrast_op, standardization_op])
to_img(_crop, "crop.png")
to_img(_left_right, "left_right.png")
to_img(_up_down, "up_down.png")
to_img(_brightness, "brightness.png")
to_img(_contrast, "contrast.png")
to_img(_std, "std.png")
for i in range(3):
    ret = sess.run(data_augment_op)
    to_img(ret, "data_aug_%i.png"%(i+1))

coord.request_stop()
coord.join(threads)
sess.close()
```

## 実行結果

```
$ python data_augument.py
```

元画像 : 

![](/img/Lenna.png)

`crop.png` : 

![](/img/crop.png)

`left_right.png` : 

![](/img/left_right.png)

`up_down.png` : 

![](/img/up_down.png)

`brightness.png` : 

![](/img/brightness.png)

`contrast.png` : 

![](/img/contrast.png)

`std.png` : 

![](/img/std.png)

`data_aug_1.png` : 

![](/img/data_aug_1.png)

`data_aug_2.png` : 

![](/img/data_aug_2.png)

`data_aug_3.png` : 

![](/img/data_aug_3.png)
