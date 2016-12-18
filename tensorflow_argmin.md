# Tensorの最小値・最大値

`tf.argmin(input, dimension, name=None)`

最小値のインデックスを返す

* `input`: `Tensor`
* `dimension`: 次元
    * ベクトルの場合、0
    * 行列の場合、0:列、1:行

`tf.argmax(input, dimension, name=None)`

最大値のインデックスを返す

* `input`: `Tensor`
* `dimension`: 次元

サンプルコード :

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# 4x4行列 要素の値は0〜100の乱数
const_data = np.random.randint(0, 100, (4,4))
const = tf.constant(const_data, dtype=tf.float32)
# 最小値 各行の最小値
argmin_op = tf.argmin(const, 1)
# 最大値 各行の最大値
argmax_op = tf.argmax(const, 1)

with tf.Session() as sess:
    c, min, max = sess.run([const,argmin_op,argmax_op])
    print c
    print min
    print max
```

実行結果 :

```
[[ 57.   6.  77.  28.]
 [ 40.  86.  85.   7.]
 [ 55.   4.  65.  49.]
 [ 65.  88.  37.  86.]]
[1 3 1 2]
[2 1 2 1]
```

## 参考

* [TensorFlow API]()
