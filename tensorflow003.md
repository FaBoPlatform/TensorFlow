
# 行列の足し算と掛け算

## Sample

```python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

sess = tf.Session()
print("a=2, b=3")
print("Add: %i" % sess.run(a+b))
print("Mult: %i" % sess.run(a*b))
```
