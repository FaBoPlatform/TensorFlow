# Tensorの集約

サンプルコード :

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

x_data = np.zeros((2,2))
x = tf.constant(x_data)
y_data = np.ones((2,2))
y = tf.constant(y_data)
z_data = np.eye(2)
z = tf.constant(z_data)

# 縦方向にTensorをまとめる
stack_op = tf.stack([x,y,z])
# 横方向にTensorをまとめる
stack_op2 = tf.stack([x,y,z], axis=1)

with tf.Session() as sess:
    rx, ry, rz = sess.run([x,y,z])
    print rx
    print ry
    print rz

    p, ps = sess.run([stack_op, tf.shape(stack_op)])
    print p
    print ps
    p, ps = sess.run([stack_op2, tf.shape(stack_op2)])
    print p
    print ps
```

実行結果 :

```
[[ 0.  0.]
 [ 0.  0.]]
[[ 1.  1.]
 [ 1.  1.]]
[[ 1.  0.]
 [ 0.  1.]]
[[[ 0.  0.]
  [ 0.  0.]]

 [[ 1.  1.]
  [ 1.  1.]]

 [[ 1.  0.]
  [ 0.  1.]]]
[3 2 2]
[[[ 0.  0.]
  [ 1.  1.]
  [ 1.  0.]]

 [[ 0.  0.]
  [ 1.  1.]
  [ 0.  1.]]]
[2 3 2]
```

## 参考

* [TensorFlow API](https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining)
