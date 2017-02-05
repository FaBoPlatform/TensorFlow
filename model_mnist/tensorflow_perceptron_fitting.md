# パーセプトロンによるフィッティング

範囲-1〜1のcos関数の値に正規分布の乱数によるノイズを混ぜたデータをパーセプトロンによりフィッティングする

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

# 9次多項式による多項式フィッティング(パーセプトロン版)
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
# ラベル(正解データ)
t = tf.placeholder(tf.float32, shape=(None,1))

def single_layer(input_layer, node_num=1024):
    """隠れ層"""
    w = tf.Variable(tf.truncated_normal([dim,node_num]))
    b = tf.Variable(tf.zeros([node_num]))
    f = tf.matmul(input_layer, w) + b
    layer = tf.nn.relu(f)
    return layer

def output_layer(layer, node_num=1024):
    """出力層"""
    w = tf.Variable(tf.zeros([node_num,1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(layer, w) + b
    return y

# 隠れ層
hidden_layer = single_layer(X, node_num=64)
# 出力層
y = output_layer(hidden_layer, node_num=64)

# 誤差関数(二乗誤差)
loss = tf.reduce_mean(tf.square(y - t))
# Adam
optimizer = tf.train.AdamOptimizer()
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
[Train] step: 200, loss: 0.133793
[Train] step: 400, loss: 0.049397
[Train] step: 600, loss: 0.034507
[Train] step: 800, loss: 0.031959
[Train] step: 1000, loss: 0.031614
```

グラフ :

![](/img/tensorflow_perceptron_fitting.png)

## 参考

