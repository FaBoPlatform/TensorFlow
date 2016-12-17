
# Placeholder

> tf.placeholder(dtype, shape=None, name=None)


|引数名|概要|
|:--|:--|
| dtype | Tensorの型 |
| shape | Tensorの形状, 無指定の場合は任意の形状のTensorを渡せる |
| name | 操作の名前 |

## Sample1

```python
import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)

sess = tf.Session()
print("Add: %i" % sess.run(add, feed_dict={a:2, b:3}))
```

## Sample2

```python
import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# 4x4行列a
mat_a = np.array([[2.0, 1.0], [4.0, 2.0]])
# 4x4行列b
mat_b = np.array([[1.0, 1.0], [6.0, 3.0]]) 

add_op = tf.add(a, b)

sess = tf.Session()
print("Add: %r" % sess.run(add_op, feed_dict={a:mat_a, b:mat_b}))
```

# 参考

[Placeholder](https://www.tensorflow.org/api_docs/python/io_ops/placeholders#placeholder)

