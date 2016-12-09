
# Placeholder

> tf.placeholder(dtype, shape=None, name=None)


|引数名|概要|
|:--|:--|
| dtype | Tensorの型 |
| shape | Tensorの形状, 無指定の場合は任意の形状のTensorを渡せる |
| name | 操作の名前 |

## Sample

```python
import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)

sess = tf.Session()
print("Add: %i" % sess.run(add, feed_dict={a:2, b:3}))
```

