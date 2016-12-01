# 標準正規分布のTensorを作る

学習結果および計算結果に再現性を持たせるために使う。

`tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)`

* `shape` Tensorのサイズ
* `mean` 平均
* `stddev` 標準偏差
* `dtype` 値の型

デフォルトでは`mean=0.0, stddev=1.0`となっており、標準正規分布になっている。

```python
# coding:utf-8
import tensorflow as tf

# 乱数のシードを設定する
tf.set_random_seed(20200724)

# 標準正規分布による乱数を値に持つ4x4行列
x = tf.random_normal(shape=(4,4))

with tf.Session() as sess:
    y = sess.run(x)
    print y
```

計算結果

```shell
[[ 0.91728258  0.16987979 -0.06950273  0.41364911]
 [-0.23856264 -3.08316779  1.08747196 -0.89166135]
 [ 0.04147797 -0.35033408 -0.06985369  1.61021757]
 [ 0.68233532 -0.31947246 -0.26191279 -0.22261487]]
```

# 参考

* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/constant_op.html#random_normal)