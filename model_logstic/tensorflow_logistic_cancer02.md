# ガン評価 学習

breast-cancer-wisconsinデータセットを使い、TensorFlowによるロジスティック回帰を行う。

Sample

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TensorFlow r1.0.0
# Python 2.7.6
import numpy as np
import tensorflow as tf

# データセットを読み込む
# なお欠損値は0とした
dataset = np.genfromtxt("./breast-cancer-wisconsin.data", delimiter=',', dtype=np.uint32, filling_values=(0))

# 重複したデータを省く
_,index = np.unique(dataset[:,0], return_index=True)
dataset = dataset[index]
# 2が良性 => 0に置き換える
dataset[:,10][np.where(dataset[:,10] == 2)] = 0
# 4が悪性 => 1に置き換える
dataset[:,10][np.where(dataset[:,10] == 4)] = 1

# 患者のデータ
data = dataset[:,1:10]

# ラベル(正解データ)
labels = dataset[:,10]

# データを学習とテスト用で7:1に分割する
train_data_size = len(data) - len(data) // 8
test_data_size = len(data) // 8

# 訓練用データ
train_data = data[:train_data_size]
train_labels = labels[:train_data_size].reshape(train_data_size, 1)

# テスト用データ
test_data = data[train_data_size:]
test_labels = labels[train_data_size:].reshape(test_data_size, 1)

# Nx9行列 (Nはデータ数)
X = tf.placeholder(tf.float32, shape=(None,9))
# 9x1行列
w = tf.Variable(tf.zeros(shape=(9,1)))
w0 = tf.Variable(tf.zeros(shape=(1)))
# 行列の積
f = tf.matmul(X, w) + w0
# シグモイド関数
p = tf.sigmoid(f)

# 正解データ Nx1行列 (Nはデータ数)
t = tf.placeholder(tf.float32, shape=(None, 1))
# 値が0にならないようにする
cp = tf.clip_by_value(p,1e-10,1.0)
cnp = tf.clip_by_value(1-p, 1e-10,1.0)
cost_f = t*tf.log(cp)+(1-t)*tf.log(cnp)
# コスト関数の定義
cost = -tf.reduce_sum(cost_f)

# トレーニングアルゴリズムとしてAdamを採用
train_step = tf.train.AdamOptimizer().minimize(cost)

# モデルの予測と正解が一致するか調べる
correct_pred = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
# モデルの精度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    for _ in range(20000):
        i += 1
        # トレーニング
        sess.run(train_step, feed_dict={X:train_data,t:train_labels})
        if i % 1000 == 0:
            # コストと精度を出力
            c, acc = sess.run([cost, accuracy], feed_dict={X: train_data,t:train_labels})
            # テスト用データを使って評価
            tacc = sess.run(accuracy, feed_dict={X:test_data,t:test_labels})
            print "step: %d, cost: %f, acc: %f, test acc: %f" % (i, c, acc, tacc)
```

実行結果

```
step: 1000, cost: 173.826935, acc: 0.909734, test acc: 0.950000
step: 2000, cost: 137.149139, acc: 0.932743, test acc: 0.975000
step: 3000, cost: 111.714836, acc: 0.950442, test acc: 0.987500
step: 4000, cost: 92.708443, acc: 0.955752, test acc: 0.987500
step: 5000, cost: 78.816345, acc: 0.961062, test acc: 0.987500
step: 6000, cost: 68.754555, acc: 0.961062, test acc: 0.987500
step: 7000, cost: 61.481083, acc: 0.962832, test acc: 1.000000
step: 8000, cost: 56.248322, acc: 0.962832, test acc: 1.000000
step: 9000, cost: 52.524582, acc: 0.961062, test acc: 1.000000
step: 10000, cost: 49.924149, acc: 0.962832, test acc: 1.000000
step: 11000, cost: 48.161461, acc: 0.966372, test acc: 1.000000
step: 12000, cost: 47.021065, acc: 0.966372, test acc: 1.000000
step: 13000, cost: 46.336304, acc: 0.966372, test acc: 1.000000
step: 14000, cost: 45.972946, acc: 0.966372, test acc: 1.000000
step: 15000, cost: 45.817200, acc: 0.966372, test acc: 1.000000
step: 16000, cost: 45.771610, acc: 0.966372, test acc: 1.000000
step: 17000, cost: 45.764835, acc: 0.966372, test acc: 1.000000
step: 18000, cost: 45.764572, acc: 0.966372, test acc: 1.000000
step: 19000, cost: 45.764626, acc: 0.966372, test acc: 1.000000
step: 20000, cost: 45.764568, acc: 0.966372, test acc: 1.000000
```

## 参考

* [Breast Cancer Wisconsin (Prognostic) Data Set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Prognostic))
