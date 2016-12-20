# Tensorの比較

サンプルコード :

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

x_data = np.arange(0, 4).reshape(2, 2)
x = tf.constant(x_data, dtype=tf.float32)
y_data = np.arange(0, 4).reshape(2, 2)
y = tf.constant(y_data, dtype=tf.float32)
z_data = [[-1, 5], [2, 0]]
z = tf.constant(z_data, dtype=tf.float32)

# 比較オペレーション
equal_op = tf.equal(x, y)
not_equal_op = tf.equal(x, y)
less_op = tf.less(x, z)
greater_op = tf.greater(x, z)

with tf.Session() as sess:
    rx, ry, rz = sess.run([x,y,z])
    print rx
    print ry
    print rz
    re, rn, rl, rg = sess.run([equal_op, not_equal_op, less_op, greater_op])
    print re
    print rn
    print rl
    print rg
```

実行結果 :

```
[[ 0.  1.]
 [ 2.  3.]]
[[ 0.  1.]
 [ 2.  3.]]
[[-1.  5.]
 [ 2.  0.]]
[[ True  True]
 [ True  True]]
[[ True  True]
 [ True  True]]
[[False  True]
 [False False]]
[[ True False]
 [False  True]]
```

## 参考

* [TensorFlow API](https://www.tensorflow.org/api_docs/python/control_flow_ops/comparison_operators)
