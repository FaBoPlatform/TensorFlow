
# ロジスティック回帰

## Coding

```python
# coding:utf-8
import tensorflow as tf
import numpy as np

x_positive = np.random.randn(500,1) + 2
y_positive = np.random.randn(500,1) + 2

x_negative = np.random.randn(500,1) - 2
y_negative = np.random.randn(500,1) - 2

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

STATE = np.zeros((N*2,2))
for i in xrange(N*2):
    if i < N:
        STATE[i][0] = 1
    else:
        STATE[i][1] = 1

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'

with tf.Graph().as_default():
        
    # 変数の定義
    X = tf.placeholder(tf.float32, name = "input")
    y = tf.placeholder(tf.float32, name = "output")
    W = tf.Variable(tf.random_normal([2,2], stddev=0.01), name = "Weight")
    b = tf.Variable(tf.random_normal([2], stddev=0.01), name = "bias")

    # ロジスティック回帰のモデル
    with tf.name_scope('forward'):
        y_pred =  tf.nn.softmax(tf.matmul(X, W) + b)

    # 損失関数
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, STATE))

    # トレーニング
    learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 精度の計算
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(STATE,1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Session
    sess = tf.Session()

    # TensorBoardへの反映
    W_graph = tf.summary.histogram("W_graph", W)
    b_graph = tf.summary.histogram("b_graph", b)
    y_graph = tf.summary.histogram("y_graph", y)
    loss_graph = tf.summary.scalar("loss_graph", loss)
    
    # Summary                    
    summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    summary_op = tf.summary.merge_all()
    
    # 変数の初期化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # トレーニング回数
    training_step = 3000
    validation_step = 10
            
    # トレーニング
    for step in xrange(training_step):
        sess.run(train_op, feed_dict={X: VIRUS, y: STATE})

        if step % validation_step == 0:
            accuracy_output,loss_output = sess.run([accuracy_op,loss], feed_dict={X: VIRUS, y: STATE})
            print "step %d, loss %f, accuracy %f" % (step,loss_output,accuracy_output)

            # TensorBoardにも反映
            summary_str = sess.run(summary_op, feed_dict={X: VIRUS, y: STATE})
            summary_writer.add_summary(summary_str, step)
                         
    summary_writer.flush()
```
