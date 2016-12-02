# Tensorの総和・総責

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# 2x2行列
# [[1, 2],
#  [3, 4]])
x_data = np.arange(1, 5).reshape((2,2))

# 2x2行列のPlaceholder
x = tf.placeholder(tf.float32, shape=(2,2))

# 全要素の和
a1 = tf.reduce_sum(x)
# 列ごとの和
a2 = tf.reduce_sum(x, 0)
# 行ごとの和
a3 = tf.reduce_sum(x, 1)

# 全要素の積
b1 = tf.reduce_prod(x)
# 列ごとの積
b2 = tf.reduce_prod(x, 0)
# 行ごとの積
b3 = tf.reduce_prod(x, 1)

with tf.Session() as sess:
    print sess.run(x, feed_dict={x:x_data})
    # 総和
    print sess.run(a1, feed_dict={x:x_data})
    print sess.run(a2, feed_dict={x:x_data})
    print sess.run(a3, feed_dict={x:x_data})
    # 総乗
    print sess.run(b1, feed_dict={x:x_data})
    print sess.run(b2, feed_dict={x:x_data})
    print sess.run(b3, feed_dict={x:x_data})
```

計算結果

```shell
[[ 1.  2.]
 [ 3.  4.]]
10.0
[ 4.  6.]
[ 3.  7.]
24.0
[ 3.  8.]
[  2.  12.]
```

# 参考

* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#reduce_sum)