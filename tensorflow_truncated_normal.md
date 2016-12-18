# 切断正規分布

標準偏差の2倍の間に収まるような乱数を生成する

`tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)`

* `shape`: Tensorの形式
* `mean`: 正規分布の平均。デフォルト 0.0
* `stddev`: 正規分布の標準偏差。デフォルト 1.0
* `dtype`: 値の型

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# デフォルトは1.0なので、乱数の値は-2〜2の間に収まる
truncated_normal = tf.truncated_normal([20])
with tf.Session() as sess:
    val = sess.run(truncated_normal)
    print val
```

## 参考

* [切断正規分布の解説](https://ja.wikipedia.org/wiki/%E5%88%87%E6%96%AD%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83)
