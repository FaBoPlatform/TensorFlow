
# Placeholder

## Sample

```python
import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
                print("Add: %i" % sess.run(add, feed_dict={a: 2, b:3}))
                print("Mult: %i" % sess.run(mul, feed_dict={a: 2, b:3}))
```
