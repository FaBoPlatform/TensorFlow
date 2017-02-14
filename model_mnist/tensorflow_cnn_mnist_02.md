# MNIST 畳込みニューラルネットワーク

[Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)

ネットワークの構成 : 

```
入力層 => 
畳込み層 => プーリング層 => 
畳込み層 => プーリング層 => 
全結合層(ドロップアウト付) => 
出力層(ソフトマックス関数)
```

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
    return tf.Variable(initial,trainable=True)

def bias_variable(shape):
    """初期化済みのバイアス"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,trainable=True)

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

### 交差エントロピーコスト関数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y_conv))
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
sess.run(tf.global_variables_initializer())
i = 0
for _ in range(20000):
    i += 1
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={X:batch[0],t:batch[1],keep_prob:0.5})
    if i % 500 == 0:
        train_acc, train_loss = sess.run([accuracy,loss], feed_dict={X:batch[0],t:batch[1],keep_prob:1.0})
        print "[Train] step: %d, loss: %f, acc: %f" % (i, train_loss, train_acc)

# テストデータによるモデルの評価
# ドロップアウトを適用しないため、keep_prob:1.0
test_acc = sess.run(accuracy, feed_dict={X:mnist.test.images,t: mnist.test.labels,keep_prob:1.0})
print "[Test] acc: %f" % test_acc

sess.close()
```

実行結果 :

```
```

## 参考

* [Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)
