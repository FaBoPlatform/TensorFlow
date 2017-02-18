# 3クラス分類 Tensorboard編

Tensorboardによりトレーニングの様子を可視化する。
先にIRISデータ作成を実行し、bazdekIris.dataを作成しておくこと。

サンプルコード :

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TensorFlow r1.0.0
# Python 2.7.6
"""
irisデータセットを使った3クラス分類
"""
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

# 隠れ層のノード数
node_num = 1024
w_hidden = tf.Variable(tf.truncated_normal([4,node_num]))
b_hidden = tf.Variable(tf.zeros([node_num]))
f_hidden = tf.matmul(X, w_hidden) + b_hidden
hidden_layer = tf.nn.relu(f_hidden)

# 出力層
w_output = tf.Variable(tf.zeros([node_num,3]))
b_output = tf.Variable(tf.zeros([3]))
f_output = tf.matmul(hidden_layer, w_output) + b_output
p = tf.nn.softmax(f_output)

# 交差エントロピー
cross_entropy = t * tf.log(p)
# 誤差関数
loss = -tf.reduce_mean(cross_entropy)
# トレーニングアルゴリズム
# 勾配降下法 学習率0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)
# モデルの予測と正解が一致しているか調べる
correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
# モデルの精度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


### 学習の実行
with tf.Session() as sess:
    # ログの設定
    tf.summary.histogram("Hidden_layer_wights", w_hidden)
    tf.summary.histogram("Hidden_layer_biases", b_hidden)
    tf.summary.histogram("Output_layer_wights", w_output)
    tf.summary.histogram("Output_layer_wights", b_output)
    tf.summary.scalar("Accuracy", accuracy)
    tf.summary.scalar("Loss", loss)
    summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter("./iris_cassification_log", sess.graph)
    sess.run(tf.global_variables_initializer())

    i = 0
    for _ in range(2000):
        i += 1
        # トレーニング
        sess.run(train_step, feed_dict={X:train_data,t:train_labels})
        # 200ステップごとに精度を出力
        if i % 200 == 0:
            # コストと精度を出力
            train_summary, train_loss, train_acc = sess.run([summary, loss, accuracy], feed_dict={X:train_data,t:train_labels})
            writer.add_summary(train_summary, i)
            print "Step: %d" % i
            print "[Train] cost: %f, acc: %f" % (train_loss, train_acc)
```

実行結果 :

```
$ tensorboard  --logdir=./iris_cassification_log
```

誤差と精度の推移 :

![tensorboard_iris_01](/img/tensorboard_iris_01.jpg)

重みとバイアスの分布 :

![tensorboard_iris_02](/img/tensorboard_iris_02.jpg)
