
# TensorFlowのインストール

OS XでのTensorFlowのインストール
https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

```shell
# Mac OS X, CPU only, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py2-none-any.whl

# Mac OS X, GPU enabled, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.11.0-py2-none-any.whl
```

```shell
# Python 2
$ sudo pip install --upgrade $TF_BINARY_URL
```

0.11.0を使うと
```shell
ImportError: No module named _pywrap_tensorflow
```
のエラーが出てしまう。その場合は、

0.10.0を使う
```shell
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl
```

# Sample1

```python
import tensorflow as tf

hello = tf.constant('Hello')
sess = tf.Session()
print sess.run(hello)
```

# Sample2

```python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
                print("a=2, b=3")
                print("Add: %i" % sess.run(a+b))
                print("Mult: %i" % sess.run(a*b))
```
# Sample3

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

# Sample4

```python
import tensorflow as tf

# [3,3]
matrix1 = tf.constant([[3,3.]])
# 2x1 matrix
# [2]
# [2]
matrix2 = tf.constant([[2.],[2.]])

print matrix1
print matrix2

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
                # 3 * 2 + 3 * 2 = 12
                result = sess.run(product)
                print(result)
```

# Sample5

```python
import tensorflow as tf
import numpy as np

matX = tf.placeholder(tf.float32, [None, 2])
matY = tf.placeholder(tf.float32, [None, 3])

train_op = tf.matmul(matX, matY)

data1 = np.array([1,1])
data1 = data1.reshape(1,2)
data2 = np.array([2,2,2])

sess = tf.Session()
sess.run(train_op, feed_dict={matX:data1, matY:data2})
```

