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

g = tf.Graph()

with g.as_default():
    # Create the model
    x = tf.placeholder(tf.float32, shape=(None, 4))
    W = tf.Variable(tf.zeros([4, 3]), name="vaiable_W")
    b = tf.Variable(tf.zeros([3]), name="variable_b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    print y
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, shape=(None,3))
    cross_entropy = y_ * tf.log(y)
    loss = -tf.reduce_sum(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    sess = tf.Session()

    # Train
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(2000):
      sess.run(train_step, feed_dict={x:train_data,y_:train_labels})

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # モデルの精度
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(accuracy.eval({x:train_data,y_:train_labels}, sess))

# Store variable
_W = W.eval(sess)
_b = b.eval(sess)

sess.close()

# Create new graph for exporting
g_2 = tf.Graph()
with g_2.as_default():
    # Reconstruct graph
    x_2 = tf.placeholder(tf.float32, shape=(None,4), name="input")
    W_2 = tf.constant(_W, name="constant_W")
    b_2 = tf.constant(_b, name="constant_b")
    y_2 = tf.nn.softmax(tf.matmul(x_2, W_2) + b_2, name="output")
    print y_2
    sess_2 = tf.Session()

    init_2 = tf.global_variables_initializer()
    sess_2.run(init_2)


    graph_def = g_2.as_graph_def()

    tf.train.write_graph(graph_def, './tmp/iris-practice',
                         'iris-graph-notLayer.pb', as_text=False)

    # Test trained model
    y__2 = tf.placeholder(tf.float32, [None, 3])
    correct_prediction_2 = tf.equal(tf.argmax(y_2, 1), tf.argmax(y__2, 1))
    accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
    print(accuracy_2.eval({x_2:train_data,y__2:train_labels}, sess_2))

```

学習で得た重みとバイアスの値をグラフに書き込むため、学習した後に再度グラフを生成している  

実行結果 :

```
Tensor("Softmax:0", shape=(?, 3), dtype=float32)
0.983333
Tensor("output:0", shape=(?, 3), dtype=float32)
0.983333
```
