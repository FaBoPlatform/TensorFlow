# 標準正規分布のTensorを作る

正規分布により乱数を生成する。

> tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)


|変数|概要|
|:--|:--|
|shape|Tensorのサイズ|
|mean|平均|
|stdev|標準偏差| 
|dtype|値の型|
|seed|シード|
|name|操作名|

デフォルトでは`mean=0.0, stddev=1.0`となっており、標準正規分布になっている。

# Sample

```python
# coding:utf-8
import tensorflow as tf

# 標準正規分布による乱数を値に持つ3x3行列
x = tf.random_normal(shape=(3,3))

sess = tf.Session()
y = sess.run(x)
print y
```

計算結果

```shell
[[ 1.09942019  0.08562929  0.03443986]
 [-0.73919928 -0.21810924  0.91688985]
 [ 0.50970089  0.08562437  0.54271621]]
```

# 実験1
正規分布の要素を1万個に増やし、平均値と標準偏差を確認する。

```python
# coding:utf-8
import tensorflow as tf
import numpy as np

# 標準正規分布による乱数を値に持つ3x3行列
x = tf.random_normal(shape=(5000,2))

sess = tf.Session()
y = sess.run(x)

print np.average(y)
print np.std(y)
```

出力結果
```shell
0.0047531
1.00467
```

要素数を増やせば増やすほど、平均値0、標準偏差 1に近づいていく。

# 実験2
平均値と標準偏差を指定して正規分布を作成する。

```python
# coding:utf-8
import tensorflow as tf
import numpy as np

# 標準正規分布による乱数を値に持つ3x3行列
x = tf.random_normal(shape=(5000,2), mean=1, stddev=10)

sess = tf.Session()
y = sess.run(x)

print np.average(y)
print np.std(y)
```

出力結果
```shell
0.977841
10.0514
```

要素数を増やせば増やすほど、指定した平均値と標準偏差に近づいていく。

# 参考

* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/constant_op.html#random_normal)