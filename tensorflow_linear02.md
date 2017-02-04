# 線形回帰 TensorBoard1

## ニューラルネットワークのグラフ化

まず最初に、ニューラルネットワークのグラフ化をTensorBoardを用いておこないます。ニューラルネットワークのグラフ化には、`tf.Graph()`を用います。Dataの保存先と、グラフ化した箇所を`with tf.Graph().as_default():`で囲い、`tf.summary.FileWriter()`で保存先フォルダを指定し、最後に`summary_writer.flush()`で書き込みます。

```pyhton

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'

...

# セッション
with tf.Session() as sess:

	...

	# Summary
	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

	with tf.Graph().as_default():

	...

	summary_writer.flush()	

```

## Coding

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

b_train = -1
w_train = 0.7
x_train = np.linspace(0, 1.0, 100)
y_train = x_train * w_train + b_train

plt.figure(1)
plt.plot(x_train, y_train, 'ro', label='Data')
plt.plot(x_train, y_train, 'k-', label='Line')
plt.show()

# GraphのReset(TF関連処理の一番最初に呼び出す)
tf.reset_default_graph()

# 変数の定義
x = tf.placeholder(tf.float32, name="input")
y = tf.placeholder(tf.float32, name="output")
w = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# 線形回帰のモデル
y_pred = tf.add(tf.multiply(x, w), b)

# 損失関数
loss = tf.reduce_mean(tf.pow(y_pred - y, 2))

# Optimizer
# 勾配降下法
learning_rate = 0.1
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Data保存先
LOGDIR = './data'

# セッション  
with tf.Session() as sess:
  # 変数の初期化
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  # Summary
  summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    
  # Graph
  with tf.Graph().as_default() as graph:
  
    # トレーニング回数
    training_step = 1500
    validation_step = 100

    #トレーニング
    for step in range(training_step):
      sess.run(train_op, feed_dict={x:x_train, y:y_train})

      # 途中経過表示
      if step % validation_step == 0:
        loss_output = sess.run(loss, feed_dict={x:x_train, y:y_train})
        w_output = sess.run(w)
        b_output = sess.run(b)
        print "Step %i, cost %f, weight %f, bias %f" % (step, loss_output, w_output, b_output)

  summary_writer.flush()
```

## TensorBoardの起動

![](/img/tensorboard01.png)

![](/img/tensorboard02.png)

```
!tensorboard --logdir=data/ 
```
を実行します。TensorBoardがport 6006で起動します。　localhost:6006に接続するとTensorBoardが表示されます。

アルゴリズムが表示されます。

![](/img/linear007.png)

## TensorBoardの終了

Datalab上では、TensorBoardがForegroundで起動しっぱなしになるので、セッションをResetし、Foregroundでの起動を停止します。

![](/img/tensorboard03.png)

![](/img/tensorboard04.png)

## グラフを見やすくする

活性化関数と損失関数の箇所を`with tf.name_scope('名前'):`で囲います。こうする事で、Graphの箇所が囲われて見やすくなります。

```python
	#活性化関数
	with tf.name_scope('forward'):
        y_pred = tf.add(tf.mul(X, W), b)
    
    # 損失関数
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow(y_pred - y, 2))
```

TensorBoardに反映させるには、本サンプルを実行し、dataフォルダのデータが更新されたのちに、再びtensorboardを起動する。

```
!tensorboard --logdir=data/ 
```

![](/img/linear006.png)

![](/img/linear008.png)

![](/img/linear009.png)

![](/img/linear010.png)

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/linear_regression02.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/linear_regression02.ipynb)