# 3クラス分類 PBファイル作成

irisデータセットを使った3クラス分類の学習モデルのpbファイルを作成する  

サンプルコード :
```python
# coding:utf-8
# TensorFlow r1.0.0
# Python 2.7.6
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

g = tf.Graph()

with g.as_default():
    # Create the model
    x = tf.placeholder(tf.float32, shape=(None, 4))
    W = tf.Variable(tf.zeros([4, 3]), name="vaiable_W")
    b = tf.Variable(tf.zeros([3]), name="variable_b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, shape=(None,3))
    cross_entropy = y_ * tf.log(y)
    loss = -tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # モデルの精度
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    sess = tf.Session()

    # Train
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(2000):
      sess.run(train_step, feed_dict={x:train_data,y_:train_labels})
      if i % 200 == 0:
        # コストと精度を出力
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x:train_data,y_:train_labels})
        # テスト用データを使って評価
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x:test_data,y_:test_labels})
        print "Step: %d" % i
        print "[Train] cost: %f, acc: %f" % (train_loss, train_acc)
        print "[Test] cost: %f, acc: %f" % (test_loss, test_acc)

    print(accuracy.eval({x:train_data,y_:train_labels}, sess))

# Store variable
_W = W.eval(sess)
_b = b.eval(sess)

sess.close()

g_2 = tf.Graph()
with g_2.as_default():
    x_2 = tf.placeholder(tf.float32, shape=(None,4), name="input")
    W_2 = tf.constant(_W, name="constant_W")
    b_2 = tf.constant(_b, name="constant_b")
    y_2 = tf.nn.softmax(tf.matmul(x_2, W_2) + b_2, name="output")
    sess_2 = tf.Session()

    init_2 = tf.global_variables_initializer()
    sess_2.run(init_2)


    graph_def = g_2.as_graph_def()

    tf.train.write_graph(graph_def, './tmp/iris-practice',
                         'iris-graph.pb', as_text=False)

```

学習で得た重みとバイアスの値をグラフに書き込むため、学習した後に再度グラフを生成している  

実行結果 :

```
Step: 0
[Train] cost: 0.366078, acc: 0.366667
[Test] cost: 0.366079, acc: 0.200000
Step: 200
[Train] cost: 0.348545, acc: 0.366667
[Test] cost: 0.348238, acc: 0.200000
Step: 400
[Train] cost: 0.335609, acc: 0.391667
[Test] cost: 0.334272, acc: 0.200000
Step: 600
[Train] cost: 0.323969, acc: 0.733333
[Test] cost: 0.321557, acc: 0.600000
Step: 800
[Train] cost: 0.313308, acc: 0.866667
[Test] cost: 0.309940, acc: 0.833333
Step: 1000
[Train] cost: 0.303530, acc: 0.916667
[Test] cost: 0.299337, acc: 0.933333
Step: 1200
[Train] cost: 0.294558, acc: 0.975000
[Test] cost: 0.289659, acc: 0.933333
Step: 1400
[Train] cost: 0.286318, acc: 0.983333
[Test] cost: 0.280818, acc: 0.933333
Step: 1600
[Train] cost: 0.278742, acc: 0.975000
[Test] cost: 0.272732, acc: 1.000000
Step: 1800
[Train] cost: 0.271767, acc: 0.975000
[Test] cost: 0.265324, acc: 1.000000
0.966667
```
