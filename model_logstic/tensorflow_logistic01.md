# ロジスティック回帰 ウィルス分布

## 分布の作成

ウィルスの感染分布を作成します。赤が感染済み、青が非感染で正規部分布をx,y座標ともに2づつずらして分布を分散させます。

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_positive = np.random.randn(500,1) + 2
y_positive = np.random.randn(500,1) + 2
x_negative = np.random.randn(500,1) - 2
y_negative = np.random.randn(500,1) - 2

plt.figure(1)
plt.plot(x_positive, y_positive, 'ro', label='Data')
plt.plot(x_negative, y_negative, 'bo', label='Data')
plt.show()
```

![](/img/logistic001.png)

## 教師データの作成

次に、現在 1x500の行列である、x_positive,y_positiveとx_negative,y_negativeを2x1000の1つの配列であるVIRUSに格納します。また、感染状況を表す行列 STATEを2x1000行列で作成し、感染している場合は[1,0], 感染していない場合は、[0, 1]を代入します。

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
```


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
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, STATE))

    # トレーニング
    learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 精度の計算
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(STATE,1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Session
    sess = tf.Session()

    # TensorBoardへの反映
    W_graph = tf.summary.histogram("W_graph", W)
    b_graph = tf.summary.histogram("b_graph", b)
    y_graph = tf.summary.histogram("y_graph", y)
    cost_graph = tf.summary.scalar("cost_graph", cost)
    
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
            accuracy_output,cost_output = sess.run([accuracy_op,cost], feed_dict={X: VIRUS, y: STATE})
            print "step %d, cost %f, accuracy %f" % (step,cost_output,accuracy_output)

            # TensorBoardにも反映
            summary_str = sess.run(summary_op, feed_dict={X: VIRUS, y: STATE})
            summary_writer.add_summary(summary_str, step)
                         
    summary_writer.flush()
```

## TensorBoard

![](/img/logistic002.png)

![](/img/logistic003.png)

![](/img/logistic004.png)

![](/img/logistic005.png)

![](/img/logistic006.png)

![](/img/logistic007.png)