# 標準正規分布のTensorを作る

正規分布により乱数を生成する。

> tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)


|変数|概要|
|:--|:--|
|shape|Tensorのサイズ|
|mean|平均|
|stdev|標準偏差| 
|dtype|値の型|
|seed|シード|
|name|操作名|

デフォルトでは`mean=0.0, stddev=1.0`となっており、標準正規分布になっている。

## Sample

正規分布

![](/img/normal01.png)

1万件に増やし、正規分布が1.0に近づくかを確認　

![](/img/normal02.png)

stddevに10指定し、平均値が10に近づく事を確認

![](/img/normal03.png)

## 参考

* [切断正規分布の解説](https://ja.wikipedia.org/wiki/%E5%88%87%E6%96%AD%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83)
* https://www.tensorflow.org/versions/r1.0/api_docs/python/constant_op.html#random_normal

# 切断正規分布

標準偏差の2倍の間に収まるような乱数を生成する

`tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)`

* `shape`: Tensorの形式
* `mean`: 正規分布の平均。デフォルト 0.0
* `stddev`: 正規分布の標準偏差。デフォルト 1.0
* `dtype`: 値の型

## Sample

切断正規分布

![](/img/normal04.png)

## 参考

* [切断正規分布の解説](https://ja.wikipedia.org/wiki/%E5%88%87%E6%96%AD%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83)

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/normal.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/normal.ipynb)

