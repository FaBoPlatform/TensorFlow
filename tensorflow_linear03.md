# 線形回帰 TensorBoard2

## 経過数値のグラフ化


```pyhton

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'
...
# TensorBoardへの反映
w_graph = tf.summary.scalar("w_graph", w)
b_graph = tf.summary.scalar("b_graph", b)
y_graph = tf.summary.histogram("y_graph", y)
loss_graph = tf.summary.scalar("loss_graph", loss)

...

# セッション  
with tf.Session() as sess:

	...

	# Summary
 	summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
	summary_op = tf.summary.merge_all() 

	# Graph
  	with tf.Graph().as_default() as graph:
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
with tf.name_scope('forward'):
  y_pred = tf.add(tf.multiply(x, w), b)

# 損失関数
with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.pow(y_pred - y, 2))

# Optimizer
# 勾配降下法
learning_rate = 0.1
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Data保存先
LOGDIR = './data'

# TensorBoardへの反映
w_graph = tf.summary.scalar("w_graph", w)
b_graph = tf.summary.scalar("b_graph", b)
y_graph = tf.summary.histogram("y_graph", y)
loss_graph = tf.summary.scalar("loss_graph", loss)
  
# セッション  
with tf.Session() as sess:
  # 変数の初期化
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  # Summary
  summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
  summary_op = tf.summary.merge_all() 

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
        
        # TensorBoardにも反映
        summary_str = sess.run(summary_op, feed_dict={x:x_train, y:y_train})                         
        summary_writer.add_summary(summary_str, step)

  summary_writer.flush()
```

## TensorBoardの起動

Datalab上でtensorboardを起動します。うまく起動しない場合、Reset sessionをでResetを選び、NotebookもBrowserのReloadボタンで再起動します。

> !tensorboard --logdir=data/

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
## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/linear_regression03.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/linear_regression03.ipynb)