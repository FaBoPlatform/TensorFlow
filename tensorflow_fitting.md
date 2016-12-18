# 多項式フィッティング

範囲-1〜1のcos関数の値に正規分布の乱数によるノイズを混ぜたデータを多項式フィッティングする

サンプルコード :

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def multiplier_list(v, n):
    """
    引数を0乗からn乗したリストを返す
    例:
    multiplier_list(2, 4)
    [1, 2, 4, 8]
    """
    return [v ** i for i in range(0, n)]

# 9次多項式によるフィッティング
dim = 9
x_data = np.arange(-1.0, 1.0, 0.01)
formed_x_data = np.array([multiplier_list(x, dim) for x in x_data])
# ノイズ
bias = np.random.normal(scale=0.2, size=x_data.shape)
# cos関数にノイズをのせる
y_data = y = np.cos(2.*np.pi*x_data) + bias
# 縦ベクトルに変換
formed_y_data = y_data[:,np.newaxis]

# N x dim 行列(Nはデータ数)
X = tf.placeholder(tf.float32, shape=(None,dim))
# dim x 1 行列
w = tf.Variable(tf.zeros([dim,1]))
# 9次多項式
y = tf.matmul(X, w)
# ラベル(正解データ)
t = tf.placeholder(tf.float32, shape=(None,1))

# 誤差関数(二乗誤差)
loss = tf.reduce_mean(tf.square(y - t))
# 勾配降下法(学習率:0.15)
optimizer = tf.train.GradientDescentOptimizer(0.15)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    i = 0
    for _ in range(1000):
        i += 1
        sess.run(train_step, feed_dict={X:formed_x_data,t:formed_y_data})
        if i % 200 == 0:
            train_loss = sess.run(loss, feed_dict={X:formed_x_data,t:formed_y_data})
            print "[Train] step: %d, loss: %f" % (i,train_loss)
            # 予測値のプロット
            predict_y = sess.run(y, feed_dict={X:formed_x_data})
            plt.plot(x_data, predict_y, label="STEP %d" % i)

    # 学習データのプロット
    plt.scatter(x_data, y_data)
    plt.legend()
    plt.show()
```

実行結果 :

```
[Train] step: 200, loss: 0.302734
[Train] step: 400, loss: 0.265032
[Train] step: 600, loss: 0.250256
[Train] step: 800, loss: 0.240955
[Train] step: 1000, loss: 0.233153
```

グラフ :

学習を繰り返すごとに元のデータに近づいていることが分かる

![](/img/tensorflow_fitting.png)

## 参考

* [多項式フィッティングの行列形式](http://mathtrain.jp/leastsquarematrix)
