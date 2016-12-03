# Tensorの型・次元数・ランク・サイズ

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# 標準正規分布による乱数を持つ3x3行列
x_data = np.random.randn(3, 3)
# 3x3のTensor
x = tf.constant(x_data, shape=(3,3))

with tf.Session() as sess:
    print sess.run(x)
    # 型
    print x.dtype
    # 次元数
    print sess.run(tf.shape(x))
    # ランク
    print sess.run(tf.rank(x))
    # サイズ
    print sess.run(tf.size(x))
```

出力結果

```shell
[[-0.97490399 -0.3510322   0.90722126]
 [-0.14026331  0.10728459 -1.98280042]
 [-0.4501969  -1.5605096   0.90494265]]
<dtype: 'float64'>
[3 3]
2
9
```

## 参考

* [TensorFlow docs](https://www.tensorflow.org/versions/master/resources/dims_types.html)
