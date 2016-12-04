
# Tensor(行列)の要素の和

## tf.add

> tf.add(x, y, name=None)

|変数|概要|
|:--|:--|
|x|half, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128, stringの型の値を引数で渡せる|
|y|half, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128, stringの型の値を引数で渡せる|
|name|操作の名前(任意)|

## Sample

> mat1 + mat2 = result_mat

![](/img/tf_add.png)

```python
import tensorflow as tf 
import numpy as np

# 2 x 2行列のPlaceholder
x = tf.placeholder(tf.float32, shape=(2,2))
y = tf.placeholder(tf.float32, shape=(2,2))

# [[1,2],
#  [3,4]]
mat1 = np.arange(1,5).reshape(2,2)
# [[11, 12],
#  [13,14]]
mat2 = np.arange(11,15).reshape(2,2)

# 和のオペレーション
add_op1 = tf.add(x, y, name="add_op1")

# セッション
sess = tf.Session()

# 実行
result_mat = sess.run(add_op1, feed_dict={x:mat1, y:mat2})

# 結果を表示
print result_mat
```

結果

```shell
[[ 12.  14.]
 [ 16.  18.]]
```

# 参考

https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#add


