# モーメンタム

`tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False)`

* `learning_rate`: 学習率
* `momentum`: モーメンタム

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

### データの準備
# データセットの読み込み
dataset = np.genfromtxt("./bezdekIris.data", delimiter=',', dtype=[float, float, float, float, "S32"])
# データセットの順序をランダムに並べ替える
np.random.shuffle(dataset)

def get_labels(dataset):
    """ラベル(正解データ)を1ofKベクトルに変換する"""
    raw_labels = [item[4] for item in dataset]
    labels = []
    for l in raw_labels:
        if l == "Iris-setosa":
            labels.append([1.0,0.0,0.0])
        elif l == "Iris-versicolor":
            labels.append([0.0,1.0,0.0])
        elif l == "Iris-virginica":
            labels.append([0.0,0.0,1.0])
    return np.array(labels)

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[:4] for item in dataset]
    return np.array(raw_data)

# ラベル
labels = get_labels(dataset)
# データ
data = get_data(dataset)
# 訓練データとテストデータに分割する
# 訓練用データ
train_labels = labels[:120]
train_data = data[:120]
# テスト用データ
test_labels = labels[120:]
test_data = data[120:]


### モデルをTensor形式で実装

# ラベルを格納するPlaceholder
t = tf.placeholder(tf.float32, shape=(None,3))
# データを格納するPlaceholder
X = tf.placeholder(tf.float32, shape=(None,4))

def single_layer(X):
    """隠れ層"""
    node_num = 1024
    w = tf.Variable(tf.truncated_normal([4,node_num]))
    b = tf.Variable(tf.zeros([node_num]))
    f = tf.matmul(X, w) + b
    layer = tf.nn.relu(f)
    return layer

def output_layer(layer):
    """出力層"""
    node_num = 1024
    w = tf.Variable(tf.zeros([node_num,3]))
    b = tf.Variable(tf.zeros([3]))
    f = tf.matmul(layer, w) + b
    p = tf.nn.softmax(f)
    return p

# 隠れ層
hidden_layer = single_layer(X)
# 出力層
p = output_layer(hidden_layer)

# 交差エントロピー
cross_entropy = t * tf.log(p)
# 誤差関数
loss = -tf.reduce_mean(cross_entropy)
# トレーニングアルゴリズム
# モーメンタムを使った勾配降下法 学習率:0.001 モーメンタム:0.6
optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.6)
train_step = optimizer.minimize(loss)
# モデルの予測と正解が一致しているか調べる
correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
# モデルの精度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


### 学習の実行
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    i = 0
    for _ in range(2000):
        i += 1
        # トレーニング
        sess.run(train_step, feed_dict={X:train_data,t:train_labels})
        # 200ステップごとに精度を出力
        if i % 200 == 0:
            # コストと精度を出力
            train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X:train_data,t:train_labels})
            # テスト用データを使って評価
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X:test_data,t:test_labels})
            print "Step: %d" % i
            print "[Train] cost: %f, acc: %f" % (train_loss, train_acc)
            print "[Test] cost: %f, acc: %f" % (test_loss, test_acc)
```

実行結果 :

```
Step: 200
[Train] cost: 0.043178, acc: 0.975000
[Test] cost: 0.055161, acc: 0.966667
Step: 400
[Train] cost: 0.033914, acc: 0.975000
[Test] cost: 0.043906, acc: 0.966667
Step: 600
[Train] cost: 0.030400, acc: 0.975000
[Test] cost: 0.039310, acc: 0.966667
Step: 800
[Train] cost: 0.028546, acc: 0.975000
[Test] cost: 0.036760, acc: 0.966667
Step: 1000
[Train] cost: 0.027405, acc: 0.983333
[Test] cost: 0.035135, acc: 0.966667
Step: 1200
[Train] cost: 0.026637, acc: 0.983333
[Test] cost: 0.034016, acc: 0.966667
Step: 1400
[Train] cost: 0.026089, acc: 0.975000
[Test] cost: 0.033204, acc: 0.933333
Step: 1600
[Train] cost: 0.025680, acc: 0.975000
[Test] cost: 0.032592, acc: 0.933333
Step: 1800
[Train] cost: 0.025366, acc: 0.975000
[Test] cost: 0.032118, acc: 0.933333
Step: 2000
[Train] cost: 0.025118, acc: 0.975000
[Test] cost: 0.031744, acc: 0.933333
```

## 参考

* [モーメンタムの解説](http://postd.cc/optimizing-gradient-descent/#momentum)
