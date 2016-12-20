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
W = tf.Variable(tf.zeros([784,10]))
# バイアス
b = tf.Variable(tf.zeros([10]))
# Nx10行列
# Nはデータ数
y = tf.matmul(X,W) + b

### 交差エントロピーコスト関数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,t))
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
sess.run(tf.initialize_all_variables())
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
[Train] step: 200, loss: 0.366843, acc: 0.910000
[Train] step: 400, loss: 0.284713, acc: 0.910000
[Train] step: 600, loss: 0.221735, acc: 0.930000
[Train] step: 800, loss: 0.171403, acc: 0.960000
[Train] step: 1000, loss: 0.274226, acc: 0.930000
[Train] step: 1200, loss: 0.314422, acc: 0.910000
[Train] step: 1400, loss: 0.268113, acc: 0.910000
[Train] step: 1600, loss: 0.201741, acc: 0.950000
[Train] step: 1800, loss: 0.142961, acc: 0.950000
[Train] step: 2000, loss: 0.248565, acc: 0.920000
[Test] acc: 0.918700
```
