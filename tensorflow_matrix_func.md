# 行列関数

行列の計算を行う数学関数のサンプル

詳しくは[Matrix Math Functions](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#matrix-math-functions)を参考。

|関数名|説明|
|:-:|:-:|
|`tf.matrix_diag(x)`|ベクトルを対角行列にする|
|`tf.matrix_diag_part(x)`|対角行列をベクトルにする|
|`tf.matrix_inverse(x)`|逆行列|
|`tf.matrix_determinant(x)`|行列式|
|`tf.trace(x)`|対角和|
|`tf.matrix_transpose(x)`|転置行列|
|`tf.eye(x)`|単位行列|
|`tf.matrix_solve(x, y)`|連立方程式の解|

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# ベクトル
x_data = np.arange(1.0, 4.0)
x = tf.constant(x_data , tf.float32)
# 3x3行列
y_data = np.arange(1.0, 10.0).reshape(3, 3)
y = tf.constant(y_data , tf.float32, shape=(3,3))
# 2x3行列
z_data = np.arange(1.0, 7.0).reshape(2, 3)
z = tf.constant(z_data , tf.float32, shape=(2,3))

with tf.Session() as sess:
    print sess.run(x)
    # ベクトルを対角行列に変換
    diag = sess.run(tf.matrix_diag(x))
    print diag
    # 行列式
    # 注意: 対角行列の行列式は対角成分の積になる
    print sess.run(tf.matrix_determinant(diag))
    # 逆行列
    print sess.run(tf.matrix_inverse(diag))
    # 対角行列をベクトルに変換
    vec = sess.run(tf.matrix_diag_part(diag))
    print vec
    # 対角和 跡(対角成分の和)
    print sess.run(tf.trace(diag))
    # 転置行列
    print sess.run(y)
    print sess.run(tf.matrix_transpose(y))
    # 2x3行列の場合、3x2行列に変形される
    print sess.run(z)
    print sess.run(tf.matrix_transpose(z))
    # 3x3単位行列
    print sess.run(tf.eye(3))


# 連立方程式 :
# 2x+y+z = 15
# 4x+6y+3z = 41
# 8x+8y+9z = 83
# 解 : x=5,y=2,z=3
# 薩摩順吉, 四ツ谷晶二, "キーポイント線形代数" p.2より
a = tf.constant([[2,1,1],[4,6,3],[8,8,9]] , tf.float32, shape=(3,3))
b = tf.constant([[15],[41],[83]] , tf.float32, shape=(3,1))

with tf.Session() as sess:
    # 連立方程式
    print sess.run(tf.matrix_solve(a, b))
```

出力結果

```
[ 1.  2.  3.]
[[ 1.  0.  0.]
 [ 0.  2.  0.]
 [ 0.  0.  3.]]
6.0
[[ 1.          0.          0.        ]
 [ 0.          0.5         0.        ]
 [ 0.          0.          0.33333334]]
[ 1.  2.  3.]
6.0
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]]
[[ 1.  4.  7.]
 [ 2.  5.  8.]
 [ 3.  6.  9.]]
[[ 1.  2.  3.]
 [ 4.  5.  6.]]
[[ 1.  4.]
 [ 2.  5.]
 [ 3.  6.]]
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]
[[ 5.]
 [ 2.]
 [ 3.]]
```

## 参考

* https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#matrix-math-functions
* trace 跡 (線型代数学)
    * https://ja.wikipedia.org/wiki/%E8%B7%A1_(%E7%B7%9A%E5%9E%8B%E4%BB%A3%E6%95%B0%E5%AD%A6)
* 転置行列の解説
    * https://ja.wikipedia.org/wiki/%E8%BB%A2%E7%BD%AE%E8%A1%8C%E5%88%97
* 行列式の解説
    * https://ja.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%BC%8F
