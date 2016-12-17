# Adam

Adamは学習率を自動的に調節する

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
# Adam
optimizer = tf.train.AdamOptimizer()
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
[Train] cost: 0.017781, acc: 0.983333
[Test] cost: 0.064186, acc: 0.866667
Step: 400
[Train] cost: 0.013653, acc: 0.983333
[Test] cost: 0.071446, acc: 0.900000
Step: 600
[Train] cost: 0.010515, acc: 0.991667
[Test] cost: 0.074486, acc: 0.900000
Step: 800
[Train] cost: 0.007630, acc: 0.991667
[Test] cost: 0.077251, acc: 0.900000
Step: 1000
[Train] cost: 0.005493, acc: 0.991667
[Test] cost: 0.080875, acc: 0.900000
Step: 1200
[Train] cost: 0.004014, acc: 0.991667
[Test] cost: 0.086093, acc: 0.900000
Step: 1400
[Train] cost: 0.002870, acc: 1.000000
[Test] cost: 0.095003, acc: 0.900000
Step: 1600
[Train] cost: 0.002106, acc: 1.000000
[Test] cost: 0.102268, acc: 0.900000
Step: 1800
[Train] cost: 0.001593, acc: 1.000000
[Test] cost: 0.108177, acc: 0.900000
Step: 2000
[Train] cost: 0.001233, acc: 1.000000
[Test] cost: 0.113778, acc: 0.900000
```

## 参考

* [Adamの解説](http://postd.cc/optimizing-gradient-descent/#adam)
