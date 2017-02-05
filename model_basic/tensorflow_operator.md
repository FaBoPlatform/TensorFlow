
# Tensor(行列)の和

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

![](/img/op01.png)

## 参考

https://www.tensorflow.org/versions/r1.0/api_docs/python/math_ops.html#add

# Tensor(行列)の差

## tf.subtract

> tf.subtract(x, y, name=None)

|変数|概要|
|:--|:--|
|x|half, float32, float64, int32, int64, complex64, complex128の型の値を引数で渡せる|
|y|half, float32, float64, int32, int64, complex64, complex128の型の値を引数で渡せる|
|name|操作の名前(任意)|

## Sample

> mat2 - mat1 = result_mat

![](/img/tf_sub.png)

![](/img/op02.png)

## 参考

https://www.tensorflow.org/versions/r1.0/api_docs/python/math_ops.html#sub

# Tensor(行列)の積

## tf.matmul

> tf.matmul(x, y, name=None)

|変数|概要|
|:--|:--|
|x|half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128の型の値を引数で渡せる|
|y|half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128の型の値を引数で渡せる|
|name|操作の名前(任意)|

## Sample

> mat1 x mat2 = result_mat

![](/img/tf_matmul.png)

![](/img/op03.png)

## 参考

https://www.tensorflow.org/versions/r1.0/api_docs/python/math_ops.html#matmul

# Tensor(行列)の要素の積

## tf.multiply

> tf.multiply(mat_a, mat_b, name=None)

|変数|概要|
|:--|:--|
|x|half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128の型の値を引数で渡せる|
|y|half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128の型の値を引数で渡せる|
|name|操作の名前(任意)|

## Sample

> mat_a x mat_b = result_mat

![](/img/tf_mul.png)

![](/img/op04.png)

## 参考

https://www.tensorflow.org/versions/r1.0/api_docs/python/math_ops/arithmetic_operators#multiply

## Notebooks

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/operator.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/operator.ipynb)

