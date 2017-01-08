# 線形回帰 TensorBoard2

## 経過数値のグラフ化


```pyhton

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'

...

with tf.Graph().as_default():

	...

	# TensorBoardへの反映
	W_graph = tf.summary.scalar("W_graph", W)
    b_graph = tf.summary.scalar("b_graph", b)
    y_graph = tf.summary.histogram("y_graph", y)
	loss_graph = tf.summary.scalar("loss_graph", loss)

	...

	# Summary
	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
	summary_op = tf.summary.merge_all() 

	...

	# トレーニング
	for step in range(training_step):
    	
    	...

    	# 途中経過表示
    	if step % validation_step == 0:
        	
        	...

        	# TensorBoardにも反映
        	summary_str = sess.run(summary_op, feed_dict={X:X_train, y:y_train})                         
            summary_writer.add_summary(summary_str, step)

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
	with tf.name_scope('foward'):
		y_pred = tf.add(tf.mul(X, W), b)

	# 損失関数
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.pow(y_pred - y, 2))

	# TensorBoardへ反映する手続き
	W_graph = tf.summary.scalar("W_graph", W)
    b_graph = tf.summary.scalar("b_graph", b)
    y_graph = tf.summary.histogram("y_graph", y)
	loss_graph = tf.summary.scalar("loss_graph", loss)

	# Optimizer
	# 勾配降下法
	learning_rate = 0.5
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# セッション
	sess = tf.Session()

	# Summary
	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
	summary_op = tf.summary.merge_all() 

	# 変数の初期化
	init_op = tf.initialize_all_variables()
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

        	# TensorBoardにも反映
        	summary_str = sess.run(summary_op, feed_dict={X:X_train, y:y_train})                         
            summary_writer.add_summary(summary_str, step)    
        	
    summary_writer.flush()
```

## TensorBoardの起動

TensorBoardに反映させるには、本サンプルを実行し、dataフォルダのデータが更新されたのちに、Ctrl+CでTensorBoardのプロセスを終了し、再度

```
tensorboard --logdir=data/ --port=8080 
```

を実行し、ウェブでプレビュー>ボード上でプレビューを選択し、ブラウザを起動します。

![](/img/linear005.png)

ブラウザのSCALARとHISTGRAMSを選択します。

![](/img/linear011.png)

![](/img/linear012.png)

## 計測タイミングの調整

グラフが直角になっているので、

```python
	# トレーニング回数
	training_step = 1000
	validation_step = 100
```

の設定を、validation_stepを10に変更し、10回に1回のペースで反映するようにする。

```python
	# トレーニング回数
	training_step = 1000
	validation_step = 10
```

dataフォルダに過去のデータが残るので

```shell
rm -rf data
```

でデータを削除後、実行し、起動中のTensorBoardをCtrl+Cで停止し、再度TensorBoardを実行して表示する。

![](/img/linear013.png)