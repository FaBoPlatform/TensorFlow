
# Variable

> tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)

|引数名|概要|
|:--|:--|
| initial_value | 初期値 |
| trainable| |
| collections| |
| validate_shape| |
| caching_device| |
| name | 名前(Const) |
| variable_def| |
| dtype | Tensorの型 |
| expected_shape |  |
| import_scope| | 

## Sample

```python
import tensorflow as tf


# Create a variable.
w = tf.Variable([[1.0,1.0],[2.0,2.0]], name="name_w")

sess = tf.Session()
# Run the variable initializer.
sess.run(w.initializer)
result = sess.run(w)
print result
```


# 参考

[Valiable](https://www.tensorflow.org/api_docs/python/state_ops/variables#Variable)

