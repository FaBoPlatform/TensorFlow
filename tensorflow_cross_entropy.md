# 交差エントロピーコスト関数

交差エントロピーコスト関数は、多クラス分類の損失関数として用いられる。

`tf.nn.softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None)`

クロスエントロピーの計算を行う

* `logits`
    * 分類モデルの値
* `labels`
    * ラベルデータ
    * 3クラス分類の例: 3つ目のクラスが正解 [0.0, 0.0, 1.0]

Sample

```python
# coding:utf-8
import random
import numpy as np
import tensorflow as tf

# 乱数のシードを設定する
random.seed(20200724)

# ソフトマックス関数
# f1:y = -0.5x
# f2:y = 0.25x
# f3:y = 0.5x-5.0
# 10x1行列
x = tf.placeholder(tf.float32, shape=(10, 1))
x_data = np.arange(0.0, 10.0).reshape(10, 1)
# 1x3行列
w = tf.constant([-0.5,0.25,0.5] , tf.float32, shape=(1,3))
# 10x3行列
b = tf.constant(np.array([[0.0,0.0,-5]]*10), tf.float32, shape=(10,3))
# 10x3行列 教師(ラベル)データ
t_data = np.zeros((10,3))
# 3つのうちどれか1つを1.0にする
for row in t_data:
    row[random.randint(0, 2)] = 1.0
t = tf.constant(t_data, tf.float32, shape=(10,3))

# 行列の積 モデル
f = tf.matmul(x, w)+b

# クロスエントロピー
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(f, t)
# クロスエントロピーコスト関数
cross_entropy_loss = tf.reduce_sum(cross_entropy)

# 以下も同様
# ソフトマックス関数
p = tf.nn.softmax(f)
# クロスエントロピー
cross_entropy2 = t * tf.log(p)
# クロスエントロピーコスト関数
cross_entropy_loss2 = -tf.reduce_sum(cross_entropy2)

with tf.Session() as sess:
    print sess.run(cross_entropy_loss, feed_dict={x:x_data})
    print sess.run(cross_entropy_loss2, feed_dict={x:x_data})
```

実行結果

```
34.7221
34.7221
```

## 参考

* [交差エントロピーの解説](https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC)
* [TensorFlow API](https://www.tensorflow.org/versions/master/api_docs/python/nn.html#softmax_cross_entropy_with_logits)
