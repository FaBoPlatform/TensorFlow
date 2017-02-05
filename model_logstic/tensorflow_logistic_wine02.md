# ワイン評価

## Coding

```python
# coding:utf-8
import numpy as np
import tensorflow as tf
import pandas as pd

# データセットを読み込む
# なお欠損値は0とした
dataset = np.genfromtxt("./wine.csv", delimiter=';', dtype=np.float32, filling_values=(0))

# 重複したデータを省く
_,index = np.unique(dataset[:,0], return_index=True)
dataset = dataset[index]

# Wineのデータ
datas = dataset[:,0:11]

# Wineのラベル(品質)
labels = dataset[:,11]
N = len(labels)
vector_labels = np.zeros((N,10))
for i in xrange(N):
    vector_labels[i][int(labels[i])] = 1.0

# データを7:1で分割する
test_data_size = len(dataset) // 8

# テスト用データ
test_datas = datas[train_data_size:]
test_labels = vector_labels[train_data_size:].reshape(test_data_size, 10)

# GraphのReset
tf.reset_default_graph()

# Data保存先
LOGDIR = './data'

def single_layer(X, num_in, num_out):
    """隠れ層"""
    W = tf.Variable(tf.truncated_normal([num_in,num_out]))
    b = tf.Variable(tf.zeros([num_out]))
    f = tf.matmul(X, W) + b
    layer = tf.nn.relu(f)
    return layer

def output_layer(layer, num_in, num_out):
    """出力層"""
    W = tf.Variable(tf.zeros([num_in,num_out]))
    b = tf.Variable(tf.zeros([num_out]))
    f = tf.matmul(layer, W) + b
    p = tf.nn.softmax(f)
    return p

with tf.Graph().as_default():
    
    # 変数の定義
    X = tf.placeholder(tf.float32, shape=(None,11), name="input") # Nx11行列
    y_ = tf.placeholder(tf.float32, shape=(None,10), name="output")

    # ロジスティック回帰のモデル
    with tf.name_scope('forward'):
        hidden_layer = single_layer(X, 11, 20)
        y = output_layer(hidden_layer, 20, 10)
    
    # 損失関数
    with tf.name_scope('cost'):
        cost = -tf.reduce_sum(y_*tf.log(y))
        
    # トレーニングアルゴリズムとしてAdagradを採用
    train_op = tf.train.AdagradOptimizer(0.01).minimize(cost)
    
    # 予測
    with tf.name_scope('accuracy'):
        # モデルの予測と正解が一致しているか調べる
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # モデルの精度
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    # Session
    sess = tf.Session()

    # Summary
    summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    
    # 変数の初期化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # トレーニング回数
    training_step = 20000
    validation_step = 100

    # トレーニング
    for step in xrange(training_step):
        sess.run(train_op, feed_dict={X:test_datas, y_:test_labels})

        if step % validation_step == 0:
            accuracy_output,cost_output = sess.run([accuracy_op,cost], feed_dict={X:test_datas, y_:test_labels})
            print "step %d, cost %f, accuracy %f" % (step,cost_output,accuracy_output)

    summary_writer.flush()
```

![](/img/wine03.png)

![](/img/wine04.png)

![](/img/wine05.png)

![](/img/wine06.png)