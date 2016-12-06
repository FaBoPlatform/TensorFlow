# 平均二乗誤差

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# 乱数のシードを設定する
np.random.seed(20200724)

# 4x2行列
x = tf.placeholder(tf.float32, [4,2])
x_data = np.random.randint(0, 100, (4,2))
# 2x1行列
w = tf.constant([[0.5],[0.2]], tf.float32, shape=(2,1))
# y = xw
y = tf.matmul(x, w)
# 4x1行列
t = tf.constant([1], tf.float32, shape=(4,1))
# 二乗
square = tf.square(y-t)
# 平均二乗誤差
mse = tf.reduce_sum(square)

with tf.Session() as sess:
    square = sess.run(square, feed_dict={x:x_data})
    print square
    mse_result = sess.run(mse, feed_dict={x:x_data})
    print mse_result
```

実行結果

```
[[ 2883.69018555]
 [ 1246.08996582]
 [  718.2399292 ]
 [  198.81001282]]
5046.83
```
