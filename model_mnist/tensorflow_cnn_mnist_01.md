# MNIST 畳込みニューラルネットワーク 準備編

ネットワークの構成: 

```
入力データ → ソフトマックス関数
```

サンプルコード: 

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
# 重み 784x10行列
W = tf.Variable(tf.zeros([784,10]),trainable=True)
# バイアス
b = tf.Variable(tf.zeros([10]),trainable=True)
# Nx10行列
# Nはデータ数
y = tf.matmul(X,W) + b

### 交差エントロピーコスト関数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=t))
### 学習アルゴリズム
# 勾配降下法 学習率:0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_step = optimizer.minimize(loss)

### モデルの評価
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
# 精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### 学習の実行
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
for _ in range(2000):
    i += 1
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X:batch[0],t:batch[1]})
    if i % 200 == 0:
        train_acc, train_loss = sess.run([accuracy,loss], feed_dict={X:batch[0],t:batch[1]})
        print "[Train] step: %d, loss: %f, acc: %f" % (i, train_loss, train_acc)

# テストデータによるモデルの評価
test_acc = sess.run(accuracy, feed_dict={X:mnist.test.images,t: mnist.test.labels})
print "[Test] acc: %f" % test_acc

sess.close()
```

実行結果 :

```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
[Train] step: 200, loss: 0.407333, acc: 0.890000
[Train] step: 400, loss: 0.243457, acc: 0.910000
[Train] step: 600, loss: 0.361597, acc: 0.900000
[Train] step: 800, loss: 0.315523, acc: 0.920000
[Train] step: 1000, loss: 0.269812, acc: 0.940000
[Train] step: 1200, loss: 0.353051, acc: 0.880000
[Train] step: 1400, loss: 0.243116, acc: 0.950000
[Train] step: 1600, loss: 0.158794, acc: 0.980000
[Train] step: 1800, loss: 0.271256, acc: 0.930000
[Train] step: 2000, loss: 0.207677, acc: 0.930000
[Test] acc: 0.919700
```
