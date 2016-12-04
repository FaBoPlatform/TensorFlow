# 基本的な数学関数

Tensorを引数にとる基本的な数学関数の例

他の関数は、[TensorflowドキュメントのBasic Math Functions](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#basic-math-functions)を参考。

|関数名|説明|
|:-:|:-:|
|tf.abs(x)|絶対値|
|tf.sign(x)|符号関数|
|tf.pow(x, y)|n乗関数|
|tf.exp(x)|指数関数|
|tf.log(x)|対数関数|
|tf.sin(x)|正弦関数|
|tf.maximum(x, y)|比較して大きい値を返す|

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

x_data = np.arange(-4.0, 5.0).reshape(3, 3)
x = tf.constant(x_data , tf.float32, shape=(3,3))
# [[-4. -3. -2.]
#  [-1.  0.  1.]
#  [ 2.  3.  4.]]

with tf.Session() as sess:
    print sess.run(x)
    # 絶対値
    print sess.run(tf.abs(x))
    # 符号関数 sign function
    print sess.run(tf.sign(x))
    # n乗
    print sess.run(tf.pow(x, 1))
    # 指数関数
    print sess.run(tf.exp(x))
    # 対数関数
    print sess.run(tf.log(x))
    # 正弦関数
    print sess.run(tf.sin(x))
    # 値を比較して大きい方の値を返す
    print sess.run(tf.maximum(x, 2))
```

出力結果

```
[[-4. -3. -2.]
 [-1.  0.  1.]
 [ 2.  3.  4.]]
[[ 4.  3.  2.]
 [ 1.  0.  1.]
 [ 2.  3.  4.]]
[[-1. -1. -1.]
 [-1.  0.  1.]
 [ 1.  1.  1.]]
[[ 16.   9.   4.]
 [  1.   0.   1.]
 [  4.   9.  16.]]
[[  1.83156393e-02   4.97870669e-02   1.35335281e-01]
 [  3.67879450e-01   1.00000000e+00   2.71828175e+00]
 [  7.38905621e+00   2.00855370e+01   5.45981483e+01]]
[[        nan         nan         nan]
 [        nan        -inf  0.        ]
 [ 0.69314718  1.09861231  1.38629436]]
[[ 0.7568025  -0.14112    -0.90929741]
 [-0.84147096  0.          0.84147096]
 [ 0.90929741  0.14112    -0.7568025 ]]
[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 2.  3.  4.]
```

## 参考

* https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html
