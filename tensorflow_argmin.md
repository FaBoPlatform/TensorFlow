# 最小値・最大値

`tf.argmin(input, dimension, name=None)`

最小値のインデックスを返す

* `input`: `Tensor`
* `dimension`: 次元
    * ベクトルの場合、0
    * 行列の場合、0:列、1:行

`tf.argmax(input, dimension, name=None)`

最大値のインデックスを返す

* `input`: `Tensor`
* `dimension`: 次元

## Sample

![](/img/argmin01.png)

## 参考

* https://www.tensorflow.org/versions/r1.0/api_docs/python/math_ops/sequence_comparison_and_indexing#argmin
* https://www.tensorflow.org/versions/r1.0/api_docs/python/math_ops/sequence_comparison_and_indexing#argmax

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/argminmax.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/argminmax.ipynb)
