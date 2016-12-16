# Tensorの総和


|変数|概要|
|:--|:--|
|input_tensor|引数として渡すTensor|
|aixs|処理対象とする次元, 0 列, 1 行|
|keep_dims| trueの場合、長さ1の縮小された次元を保持。|
|name|操作の名前(任意)|

# Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# 2x2行列(Tensor)
# [[1, 2],
#  [3, 4]])
x_data = np.arange(1, 5).reshape(2,2)

# 2x2行列のPlaceholder
x = tf.placeholder(tf.float32, shape=(2,2))

# 全要素の和
a1 = tf.reduce_sum(x)
# 列ごとの和
a2 = tf.reduce_sum(x, 0)
# 行ごとの和
a3 = tf.reduce_sum(x, 1)

sess = tf.Session()
print sess.run(x, feed_dict={x:x_data})
# 総和
print sess.run(a1, feed_dict={x:x_data})
print sess.run(a2, feed_dict={x:x_data})
print sess.run(a3, feed_dict={x:x_data})
```

計算結果

```shell
[[ 1.  2.]
 [ 3.  4.]]
10.0
[ 4.  6.]
[ 3.  7.]
```

# 参考

* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#reduce_sum)
