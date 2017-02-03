# 線形回帰

線形な値の教師データから、TensorFlowのW(ウェイト)とb(バイアス)の収束を観察する。

まず、教師データは、(x,y)座標は、`y = x * W + b`で定義する。

```
b_train = -1
W_train = 0.7
X_train = np.linspace(0, 1.0, 100)
y_train = X_train * W_train + b_train
```

この線形な値の教師データを用いて、同様のWとbにTensorFlowで収束する結果を観察する。つまりTensorFlowを用いて、Wとbを導き出す。

![](/img/linear00.png)

# 教師データ

教師データの分布を確認するために、matplotlibでグラフを表示してみる。

![](/img/linear01.png)


# Coding

それでは、下記のコードで、収束を確認していく。

![](/img/linear02.png)

## 変数の定義

TensorFlowの中で扱う変数を定義する。Xにtrain_x, yにtrain_yを代入し、W, bを導き出す。

```python
x = tf.placeholder(tf.float32, name = "input")
y = tf.placeholder(tf.float32, name = "output")
w = tf.Variable(np.random.randn(), name = "weight")
b = tf.Variable(np.random.randn(), name = "bias")
```

## 活性化関数

> y_pred = X * W + b

```python
y_pred = tf.add(tf.mul(x, w), b)
```
# 損失関数

活性化関数で定義したy_predと、y(ここにはy_trainを代入)の差を2乗した平均を求めている。2乗している理由が、差が+になるか-になるか予想できないため、2乗し、すべて差を+にして、その平均値を出している。この

```python
loss = tf.reduce_mean(tf.pow(y_pred - y, 2))
```

# Optimizer

学習はOptimizerに損失関数をセットする事でおこなう。

```python
learning_rate = 0.5
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

TensorFlowで利用できるOptimizerは、下記のアルゴリズムが存在している。

|Optimizer|日本語名|
|:--|:--|
|tf.train.GradientDescentOptimizer | 勾配降下アルゴリズム |
|tf.train.AdadeltaOptimizer | Adadeltaアルゴリズム |
|tf.train.AdagradOptimizer | Adagradアルゴリズム |
|tf.train.AdagradDAOptimizer | Adagrad双対平均アルゴリズム |
|tf.train.MomentumOptimizer | Momentum(慣性項)アルゴリズム|
|tf.train.AdamOptimizer | Adamアルゴリズム |
|tf.train.FtrlOptimizer | FTRL(Follow The Reaularized Leader)アルゴリズム |
|tf.train.ProximalGradientDescentOptimizer | 近位勾配降下アルゴリズム |
|tf.train.ProximalAdagradOptimizer | 近位Adagradアルゴリズム |
|tf.train.RMSPropOptimizer | RMSpropアルゴリズム |

# セッション

```python
# セッション
with tf.Session() as sess:
```

# 変数の初期化

```python
# 変数の初期化
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
```

# 学習

```python
    # トレーニング回数
    training_step = 1000
    validation_step = 100

    # トレーニング
    for step in range(training_step):
        sess.run(train_op, feed_dict={x:x_train, y:y_train})
    
        # 途中経過表示
        if step % validation_step == 0:
            loss_output = sess.run(loss, feed_dict={x:x_train, y:y_train})
            w_output = sess.run(w)
            b_output = sess.run(b)
            print "Step %i, cost %f, weight %f, bias %f" % (step, loss_output, w_output, b_output)

```

## 結果

![](/img/linear03.png)

