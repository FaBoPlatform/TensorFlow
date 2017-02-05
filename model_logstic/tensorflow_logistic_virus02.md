#　ウィルス分布　学習

![](/img/virus201.png)

![](/img/virus202.png)

![](/img/virus203.png)

![](/img/virus204.png)

![](/img/virus205.png)

## Coding

```python
# coding:utf-8
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
LOGDIR = "./data"
 
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

with tf.Session() as sess:
  
  # 初期化処理
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  # トレーニング
  learning_rate = 0.01
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Summary
  summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

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

    summary_writer.flush()
```

## TensorBoard

![](/img/logistic002.png)

![](/img/logistic003.png)

![](/img/logistic004.png)

![](/img/logistic005.png)

![](/img/logistic006.png)

![](/img/logistic007.png)

