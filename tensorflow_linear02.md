# 線形回帰 TensorBoard1

## ニューラルネットワークのグラフ化

まず最初に、ニューラルネットワークのグラフ化をTensorBoardを用いておこないます。ニューラルネットワークのグラフ化には、`tf.Graph()`を用います。Dataの保存先と、グラフ化した箇所を`with tf.Graph().as_default():`で囲い、`tf.summary.FileWriter()`で保存先フォルダを指定し、最後に`summary_writer.flush()`で書き込みます。

```pyhton

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'

...

with tf.Graph().as_default():

	...

	# セッション
	sess = tf.Session()

	# Summary
	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

	...

	summary_writer.flush()	

```

## Coding

```python
#coding:utf-8
import tensorflow as tf
import numpy as np

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'

# この中に記述したアルゴリズムがグラフ化される
with tf.Graph().as_default():

	# トレーニングデータの作成(1x100)
	b_train = -1
	W_train = 0.7
	X_train = np.linspace(0.0, 1.0, 100)
	y_train = (X_train * W_train + b_train)

	# 変数の定義
	X = tf.placeholder(tf.float32, name = "input")
	y = tf.placeholder(tf.float32, name = "output")
	W = tf.Variable(np.random.randn(), name = "Weight")
	b = tf.Variable(np.random.randn(), name = "bias")

	# 線形回帰のモデル
	y_pred = tf.add(tf.multiply(X, W), b)

	# 損失関数
	loss = tf.reduce_mean(tf.pow(y_pred - y, 2))

	# Optimizer
	# 勾配降下法
	learning_rate = 0.5
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# セッション
	sess = tf.Session()

	# Summary
	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

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

    summary_writer.flush()
```

## TensorBoardの起動

Cloud Shellのタブ新規に立ち上げます。

![](/img/linear003.png)

作業フォルダに移動し、dataフォルダが存在しているのを確認した上で

```
tensorboard --logdir=data/ --port=8080 
```

を実行します。

![](/img/linear004.png)

ウェブでプレビュー>ボード上でプレビューを選択し、ブラウザを起動します。

![](/img/linear005.png)

ブラウザのGraphを選択します。

![](/img/linear006.png)

アルゴリズムが表示されます。

![](/img/linear007.png)

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

TensorBoardに反映させるには、本サンプルを実行し、dataフォルダのデータが更新されたのちに、Ctrl+CでTensorBoardのプロセスを終了し、再度

```
tensorboard --logdir=data/ --port=8080 
```

を実行し、ウェブでプレビュー>ボード上でプレビューを選択し、ブラウザを起動します。

![](/img/linear005.png)

ブラウザのGraphを選択します。

![](/img/linear006.png)


![](/img/linear008.png)

![](/img/linear009.png)

![](/img/linear010.png)
