# 線形回帰 その1

線形な値の教師データから、TensorFlowのW(ウェイト)とb(バイアス)の収束を観察する。

まず、教師データは、(x,y)座標は、`y = x * W + b`で定義する。

```
b_train = 0.7
W_train = -0.1
X_train = np.randmon.random((1,100))
y_train = train_X * train_W + train_b
```

この線形な値の教師データを用いて、同様のWとbにTensorFlowで収束する結果を観察する。つまりTensorFlowを用いて、Wとbを導き出す。

![](/img/linear001.png)

# 教師データ

教師データの分布を確認するために、matplotlibでグラフを表示してみる。

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

b_train = -1
W_train = 0.7
X_train = np.linspace(0, 1.0, 100)
y_train = X_train * W_train + b_train

plt.figure(1)
plt.plot(X_train, y_train, 'ro', label='Data')
plt.plot(X_train, y_train, 'k-', label='Line')
plt.show()
```

![](/img/linear002.png)


# Coding

それでは、下記のコードで、収束を確認していく。


```python
#coding:utf-8
import tensorflow as tf
import numpy as np

# トレーニングデータの作成(1x100)
b_train = -1
W_train = 0.7
X_train = np.linspace(0.0, 1.0, 100)
y_train = (X_train * W_train + b_train)

# 変数の定義
X = tf.placeholder(tf.float32, name = "input")
y = tf.placeholder(tf.float32, name = "output")
W = tf.Variable(np.random.randn(), name = "weight")
b = tf.Variable(np.random.randn(), name = "bias")

# 活性化関数
y_pred = tf.add(tf.mul(X, W), b)

# 損失関数
loss = tf.reduce_mean(tf.pow(y_pred - y, 2))

# Optimizer
# 勾配降下法
learning_rate = 0.5
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# セッション
sess = tf.Session()

# 変数の初期化
init_op = tf.global_variables_initializer()
sess.run(init_op)

# トレーニング回数
training_step = 1000
validation_step = 100

# トレーニング
for step in range(training_step):
    sess.run(train_op, feed_dict={X:X_train, y:y_train})
    
    # 途中経過表示
    if step % validation_step == 0:
        loss_output = sess.run(loss, feed_dict={X:X_train, y:y_train})
        W_output = sess.run(W)
        b_output = sess.run(b)
        print "Step %i, cost %f, weight %f, bias %f" % (step, loss_output, W_output, b_output)

```

## 変数の定義

TensorFlowの中で扱う変数を定義する。Xにtrain_x, yにtrain_yを代入し、W, bを導き出す。

```python
X = tf.placeholder(tf.float32, name = "input")
y = tf.placeholder(tf.float32, name = "output")
W = tf.Variable(np.random.randn(), name = "weight")
b = tf.Variable(np.random.randn(), name = "bias")
```

## 活性化関数

> y_pred = X * W + b

```python
y_pred = tf.add(tf.mul(X, W), b)
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
sess = tf.Session()
```

# 変数の初期化

```python
# 変数の初期化
init_op = tf.global_variables_initializer()
sess.run(init_op)
```

# 学習

```python
# トレーニング回数
training_step = 1000
validation_step = 100

# トレーニング
for step in range(training_step):
    sess.run(train_op, feed_dict={X:X_train, y:y_train})
    
    # 途中経過表示
    if step % validation_step == 0:
        loss_output = sess.run(loss, feed_dict={X:X_train, y:y_train})
        W_output = sess.run(W)
        b_output = sess.run(b)
        print "Step %i, cost %f, weight %f, bias %f" % (step, loss_output, W_output, b_output)

```

# 実行結果

Cloud MLのCloud Shellで実行します。

> python linear01.py


結果は以下のように、Wは0.7、bは-1、lossは0に収束していきます。

```
Step 0, loss 0.025229, W 0.437177, b -1.007717
Step 100, loss 0.000000, W 0.699805, b -0.999896
Step 200, loss 0.000000, W 0.700000, b -1.000000
Step 300, loss 0.000000, W 0.700000, b -1.000000
Step 400, loss 0.000000, W 0.700000, b -1.000000
Step 500, loss 0.000000, W 0.700000, b -1.000000
Step 600, loss 0.000000, W 0.700000, b -1.000000
Step 700, loss 0.000000, W 0.700000, b -1.000000
Step 800, loss 0.000000, W 0.700000, b -1.000000
Step 900, loss 0.000000, W 0.700000, b -1.000000
```

