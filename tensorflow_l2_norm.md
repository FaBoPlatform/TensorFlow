# L2正則化

サンプルコード :

```python
# coding:utf-8
import tensorflow as tf

# 乱数のシードを設定する
tf.set_random_seed(20200724)

# 入力Tensor
x = tf.random_normal([10])
# L2正則化 sum(x ** 2) / 2
l2_loss_op = tf.nn.l2_loss(x)

# 以下も同様
my_l2_loss_op = tf.reduce_sum(tf.square(x)) / 2.0

with tf.Session() as sess:
    x, l2_loss, m_l2_loss = sess.run([x, l2_loss_op, my_l2_loss_op])
    print x
    print l2_loss
    print m_l2_loss
```

実行結果 :

```
[ 0.91728258  0.16987979 -0.06950273  0.41364911 -0.23856264 -3.08316779
  1.08747196 -0.89166135  0.04147797 -0.35033408]
6.35557
6.35557
```

## 参考

* [TensofFlow API](https://www.tensorflow.org/api_docs/python/nn/losses#l2_loss)
