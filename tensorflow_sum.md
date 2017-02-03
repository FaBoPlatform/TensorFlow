# Tensorの総和


|変数|概要|
|:--|:--|
|input_tensor|引数として渡すTensor|
|aixs|処理対象とする次元, 0 列, 1 行|
|keep_dims| trueの場合、長さ1の縮小された次元を保持。|
|name|操作の名前(任意)|

## Sample

![](/img/sum01.png)

## 参考

* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#reduce_sum)

# Tensorの総積

> tf.reduce_prod(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

|変数|概要|
|:--|:--|
|input_tensor|引数として渡すTensor|
|aixs|処理対象とする次元, 0 列, 1 行|
|keep_dims| trueの場合、長さ1の縮小された次元を保持。|
|name|操作の名前(任意)|

## Sample

![](/img/prod01.png)

## 参考

* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#reduce_prod)

