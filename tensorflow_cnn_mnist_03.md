# MNIST 畳込みニューラルネットワーク 重み減衰

重み減衰は過適合を避けるために使われる

[Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)

サンプルコード :

```python
# coding:utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 結果が同じになるように、乱数のシードを設定する
tf.set_random_seed(20200724)
np.random.seed(20200724)

# MNISTデータセットの読み込み
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### モデルの実装
# ラベル(正解データ) Nx10行列
# Nはデータ数, 10はクラス数
t = tf.placeholder(tf.float32, shape=(None,10))
# 入力データ Nx784行列
# Nはデータ数, 28x28=784
X = tf.placeholder(tf.float32, shape=(None,784))

def input_image(X):
    """入力層"""
    return tf.reshape(X, [-1,28,28,1])

def weight_variable(shape):
    """初期化済みの重み"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """初期化済みのバイアス"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(X, W):
    """畳込み層"""
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(X):
    """プーリング層"""
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

### 入力層
input_layer = input_image(X)

### 畳込み層 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
### プーリング層 1
h_conv1 = tf.nn.relu(conv2d(input_layer, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### 畳込み層 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
### プーリング層 2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

### 全結合層
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

### ドロップアウト
keep_prob = tf.placeholder(tf.float32) # ドロップアウトする割合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### 出力層
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

### 正則化項 重み減衰
norm_term = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
# 正則化パラメタ
lambda_ = 0.001
### 交差エントロピーコスト関数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, t))
# コスト関数
loss = cross_entropy + lambda_ * norm_term

### 学習アルゴリズム
# Adam 学習率:0.0001
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

### モデルの評価
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(t,1))
# 精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### 学習の実行
sess = tf.Session()
sess.run(tf.initialize_all_variables())
i = 0
for _ in range(40000):
    i += 1
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={X:batch[0],t:batch[1],keep_prob:0.5})
    if i % 1000 == 0:
        train_acc, train_loss = sess.run([accuracy,loss], feed_dict={X:batch[0],t:batch[1],keep_prob:1.0})
        print "[Train] step: %d, loss: %f, acc: %f" % (i, train_loss, train_acc)
        # テストデータによるモデルの評価
        test_acc, test_loss = sess.run([accuracy,loss], feed_dict={X:mnist.test.images,t:mnist.test.labels,keep_prob:1.0})
        print "[Test] loss: %f, acc: %f" % (test_loss, test_acc)
        row = "%d,%f,%f,%f,%f\n" % (i, train_loss, train_acc, test_loss, test_acc)
        with open("evaluation.csv", "a") as fout:
            fout.write(row)

sess.close()
```

実行結果 :

```
[Train] step: 1000, loss: 9.564764, acc: 0.960000
[Test] loss: 9.496043, acc: 0.962800
[Train] step: 2000, loss: 7.332607, acc: 0.960000
[Test] loss: 7.343444, acc: 0.973200
[Train] step: 3000, loss: 5.587842, acc: 0.980000
[Test] loss: 5.605972, acc: 0.981300
[Train] step: 4000, loss: 4.164217, acc: 0.960000
[Test] loss: 4.161895, acc: 0.984500
[Train] step: 5000, loss: 3.013694, acc: 0.960000
[Test] loss: 3.012430, acc: 0.984300
[Train] step: 6000, loss: 2.119012, acc: 1.000000
[Test] loss: 2.136513, acc: 0.986900
[Train] step: 7000, loss: 1.490582, acc: 1.000000
[Test] loss: 1.522875, acc: 0.986200
[Train] step: 8000, loss: 1.118178, acc: 0.980000
[Test] loss: 1.097470, acc: 0.987600
[Train] step: 9000, loss: 0.854345, acc: 0.980000
[Test] loss: 0.810288, acc: 0.988400
[Train] step: 10000, loss: 0.598223, acc: 1.000000
[Test] loss: 0.611443, acc: 0.990100
[Train] step: 11000, loss: 0.462429, acc: 1.000000
[Test] loss: 0.483126, acc: 0.989000
[Train] step: 12000, loss: 0.362479, acc: 1.000000
[Test] loss: 0.387866, acc: 0.990300
[Train] step: 13000, loss: 0.298344, acc: 1.000000
[Test] loss: 0.322925, acc: 0.989400
[Train] step: 14000, loss: 0.259400, acc: 1.000000
[Test] loss: 0.271788, acc: 0.990300
[Train] step: 15000, loss: 0.217047, acc: 1.000000
[Test] loss: 0.235201, acc: 0.991100
[Train] step: 16000, loss: 0.190560, acc: 1.000000
[Test] loss: 0.211423, acc: 0.988800
[Train] step: 17000, loss: 0.177036, acc: 1.000000
[Test] loss: 0.184945, acc: 0.990000
[Train] step: 18000, loss: 0.145949, acc: 1.000000
[Test] loss: 0.170616, acc: 0.989400
[Train] step: 19000, loss: 0.139694, acc: 1.000000
[Test] loss: 0.153126, acc: 0.990900
[Train] step: 20000, loss: 0.130082, acc: 1.000000
```

重み減衰を使っていない場合との比較 :

![](img/weight_decay.png)

重み減衰を用いていない場合と比べると、学習が停滞しないことが分かる。
