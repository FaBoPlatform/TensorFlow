# ロジスティック回帰 学習済みデータ

# 結果の評価

予測用のOperationを用意して、[-2,-2]の座標と[2,2]の座標を渡し、ウィルスに感染しているかを評価します。

```python  
# 予測   
with tf.name_scope('predict'):
  predict_op = tf.argmax(y_pred, 1)

  ...

  # check anser
  data = [[-2,-2]]
  x_check = np.array(data)
  flag_pos = sess.run(predict_op, feed_dict={x: x_check})
  print "flag position is %d" % (flag_pos)
  data = [[2,2]]
  x_check = np.array(data)
  flag_pos = sess.run(predict_op, feed_dict={x: x_check})
  print "flag position is %d" % (flag_pos)
```

# 学習済みモデルを保存
```python  
# 学習済みモデルの保存準備
saver = tf.train.Saver()

with tf.Session() as sess:

  ...

        # TensorBoardにも反映
        summary_str = sess.run(summary_op, feed_dict={x: VIRUS, y: STATE})
        summary_writer.add_summary(summary_str, step)

        # 学習済みモデルの保存　ファイル名：checkpointとvirus-model-100.[data-00000-of-00001|index|meta] (step=100の場合)が作られる
        saver.save(sess, "virus-model", global_step=step)

    summary_writer.flush()
```

# 学習済みモデルを読み込む
```python  
# 学習済みモデルの保存準備
saver = tf.train.Saver()

with tf.Session() as sess:
  # 学習済みモデルのcheckpointファイルがあるかどうか
  ckpt = tf.train.get_checkpoint_state('./')
  if ckpt:
    # checkpointファイルから最後に保存したモデルへのパスを取得する
    last_model = ckpt.model_checkpoint_path
    print　"load " + last_model
    # 学習済みモデルを読み込む
    saver.restore(sess, last_model)
  else:
  print "initialization"
  # 初期化処理
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  # トレーニング
  learning_rate = 0.01
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  ...

```

## Coding

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TensorFlow r1.0.0
# Python 2.7.6
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_positive = np.random.randn(500, 1) + 2
y_positive = np.random.randn(500, 1) + 2
x_negative = np.random.randn(500, 1) - 2
y_negative = np.random.randn(500, 1) - 2

plt.figure(1)
plt.plot(x_positive, y_positive, 'ro', label='Data1')
plt.plot(x_negative, y_negative, 'bo', label='Data2')

N = len(x_positive)
POSITIVE = np.zeros((N,2))
for i in xrange(N):
  POSITIVE[i][0] = x_positive[i]
  POSITIVE[i][1] = y_positive[i]

NEGATIVE = np.zeros((N,2))
for i in xrange(N):
  NEGATIVE[i][0] = x_negative[i]
  NEGATIVE[i][1] = y_negative[i]

VIRUS = np.vstack([NEGATIVE, POSITIVE]).astype(np.float32)

print VIRUS

STATE = np.zeros((N*2,2), dtype=np.float32)
for i in xrange(N*2):
  if i < N:
    STATE[i][1] = 1
  else:
    STATE[i][0] = 1

print STATE

tf.reset_default_graph()
LOGDIR = "./data_virus"
 
x = tf.placeholder(tf.float32, shape=(None,2), name="input")
y = tf.placeholder(tf.float32, shape=(None,2), name="output")
w = tf.Variable(tf.random_normal([2,2], stddev=0.01), dtype=tf.float32, name="weight")
b = tf.Variable(tf.random_normal([2], stddev=0.01), dtype=tf.float32, name="bias")

# ロジスティック回帰のモデルを定義
with tf.name_scope('forward'):
  y_pred = tf.nn.softmax(tf.matmul(x,w) + b, name="forward")

# コストの計算
with tf.name_scope('cost'):
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred)
  cost = tf.reduce_mean(loss, 0)

# 精度の計算
with tf.name_scope('accuracy'):
  correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(STATE,1))
  accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 予測
with tf.name_scope('predict'):
  predict_op = tf.argmax(y_pred, 1)

# TensorBoardへの反映
w_graph = tf.summary.histogram("W_graph", w)
b_graph = tf.summary.histogram("b_graph", b)
y_graph = tf.summary.histogram("y_graph", y)
cost_graph = tf.summary.scalar("cost_graph", cost)

# 学習済みモデルの保存準備
saver = tf.train.Saver()

with tf.Session() as sess:
  # 学習済みモデルのcheckpointファイルがあるかどうか
  ckpt = tf.train.get_checkpoint_state('./')
  if ckpt:
    # checkpointファイルから最後に保存したモデルへのパスを取得する
    last_model = ckpt.model_checkpoint_path
    print("load {0}".format(last_model))
    # 学習済みモデルを読み込む
    saver.restore(sess, last_model)
  else:
    print("initialization")
    # 初期化処理
    init_op = tf.global_variables_initializer()
    sess.run(init_op)  

  # トレーニング
  learning_rate = 0.01
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Summary
  summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
  summary_op = tf.summary.merge_all()

  with tf.Graph().as_default():
    
    # トレーニング回数
    training_step = 1000
    validation_step = 100

    # トレーニング
    for step in xrange(training_step):
      sess.run(train_op, feed_dict={x: VIRUS, y: STATE})

      if step % validation_step == 0:
        accuracy_output,cost_output = sess.run([accuracy_op,cost], feed_dict={x: VIRUS, y: STATE})
        print "step %d, cost %f, accuracy %f" % (step,cost_output,accuracy_output)

        # TensorBoardにも反映
        summary_str = sess.run(summary_op, feed_dict={x: VIRUS, y: STATE})
        summary_writer.add_summary(summary_str, step)

        # 学習済みモデルの保存　ファイル名：checkpointとvirus-model-100.[data-00000-of-00001|index|meta] (step=100の場合)が作られる
        saver.save(sess, "virus-model", global_step=step)

    summary_writer.flush()

    # check anser
    data = [[-2,-2]]
    x_check = np.array(data)
    flag_pos = sess.run(predict_op, feed_dict={x: x_check})
    print "flag position is %d" % (flag_pos)
    data = [[2,2]]
    x_check = np.array(data)
    flag_pos = sess.run(predict_op, feed_dict={x: x_check})
    print "flag position is %d" % (flag_pos)
```

1のflagがたっている配列の場所が結果として取得できる。

![](/img/virus301.png)

> 1なら[1,0], 0なら[0,1]

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/virus03.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/virus03.ipynb)
