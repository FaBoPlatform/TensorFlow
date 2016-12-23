# MNIST 畳込みニューラルネットワーク セッションの保存・書込み

`tf.app.flags`を使い、TensorBoardのサマリーとセッションの保存先を指定する。

サンプルコード :

```python
# coding:utf-8
"""
filename: mnist_cnn.py
"""
import re
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 結果が同じになるように、乱数のシードを設定する
tf.set_random_seed(20200724)
np.random.seed(20200724)

# コマンドライン引数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("restore_session_path", None, "セッションの読み込み先")
tf.app.flags.DEFINE_string("summary_path", "./mnist_cnn_log", "TensorBoard用ログの保存先")
tf.app.flags.DEFINE_string("session_path", "./mnist_cnn_session", "セッションの保存先")

def weight_variable(shape, name):
    """初期化済みの重み"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    """初期化済みのバイアス"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

class InputLayer:
    """入力層"""
    def __init__(self, X, image_shape=None, name=None):
        with tf.name_scope(name):
            shape = tf.shape(X)
            output = tf.reshape(X, [-1,image_shape[0],image_shape[1],1])
            self.output = output

class ConvPoolLayer:
    """
    畳込み層 + プーリング層
    """
    def __init__(self, layer, shape=None, name=None):
        """
        input_layer: 入力層(画像データ群)
        shape: フィルタの縦サイズ, フィルタの横サイズ, 入力レイヤ数, 出力レイヤ数
        """
        with tf.name_scope(name):
            ### 畳込み層
            with tf.name_scope("convolutional_layer"):
                W = weight_variable(shape, "conv_layer_weights")
                b = bias_variable([shape[3]], "conv_layer_biases")
                conv = self.conv2d(layer.output, W, "conv_layer") + b
            ### プーリング層
            with tf.name_scope("max_pooling_layer"):
                h_conv = tf.nn.relu(conv)
                h_pool = self.max_pool_2x2(h_conv, "max_pooling_layer")
            self.output = h_pool
            # ログの設定
            tf.histogram_summary("%s_weights" % name, W)
            tf.histogram_summary("%s_biases" % name, b)

    def conv2d(self, X, W, name):
        """畳込み層"""
        return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    def max_pool_2x2(self, X, name):
        """プーリング層"""
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name=name)

class FullyConnectedLayer:
    """全結合層"""
    def __init__(self, layer, shape=None, dropout_rate=None, name=None):
        """
        shape: 入力レイヤの形式
        dropout_rate: ドロップアウト率
        """
        with tf.name_scope(name):
            ### 全結合層
            with tf.name_scope("fully_connected_layer"):
                W = weight_variable(shape, "fully_connected_layer_weights")
                b = bias_variable([shape[1]], "fully_connected_layer_biases")
                h_pool_flat = tf.reshape(layer.output, [-1, shape[0]])
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W) + b)
            ### ドロップアウト
            with tf.name_scope("dropout"):
                # ドロップアウトする割合
                h_fc_drop = tf.nn.dropout(h_fc, dropout_rate, name="dropout")
            self.output = h_fc_drop
            # ログの設定
            tf.histogram_summary("%s_weights" % name, W)
            tf.histogram_summary("%s_biases" % name, b)

class OutputLayer:
    """出力層"""
    def __init__(self, layer, shape=None, name=None):
        with tf.name_scope(name):
            with tf.name_scope("output_layer"):
                W = weight_variable(shape, "output_layer_weights")
                b = bias_variable([shape[1]], "output_layer_biases")
                y = tf.matmul(layer.output, W) + b
                self.output = y
            # ログの設定
            tf.histogram_summary("%s_weights" % name, W)
            tf.histogram_summary("%s_biases" % name, b)

class Optimizer:
    """最適化アルゴリズム"""
    def __init__(self, output_layer, t, name=None):
        with tf.name_scope(name):
            ### 交差エントロピーコスト関数
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer.output, t), name="loss")
            ### 学習アルゴリズム
            # Adam 学習率:0.0001
            optimizer = tf.train.AdamOptimizer(1e-4)
            self.loss = loss
            self.train_step = optimizer.minimize(loss)
            #  ログの設定
            tf.scalar_summary("loss", loss)

class Evaluator:
    """評価器"""
    def __init__(self, output_layer, t, name=None):
        with tf.name_scope("evaluator"):
            # モデルの評価
            correct_prediction = tf.equal(tf.argmax(output_layer.output,1), tf.argmax(t,1))
            # 精度
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                    name="accuracy")
            #  ログの設定
            tf.scalar_summary("accuracy", accuracy)
            self.accuracy = accuracy

if __name__ == "__main__":
    # MNISTデータセットの読み込み
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 入力データ
    X = tf.placeholder(tf.float32, shape=(None, 28*28), name="input_images")
    # ラベル(正解データ)
    t = tf.placeholder(tf.float32, shape=(None, 10), name="labels")

    # 入力層
    input_layer = InputLayer(X, image_shape=(28,28), name="input_layer")
    # 畳込み層 + プーリング層
    cnv_pool_layer1 = ConvPoolLayer(input_layer, shape=(5,5,1,32), name="cnv_pool_layer1")
    # 畳込み層 + プーリング層
    cnv_pool_layer2 = ConvPoolLayer(cnv_pool_layer1, shape=(5,5,32,64), name="cnv_pool_layer2")
    # ドロップアウト率
    keep_prob = tf.placeholder(tf.float32)
    # 全結合層
    fully_connected_layer = FullyConnectedLayer(cnv_pool_layer2, shape=(7*7*64,1024), dropout_rate=keep_prob, name="fully_connected_layer")
    # 出力層
    output_layer = OutputLayer(fully_connected_layer, shape=(1024,10), name="output_layer")
    # 最適化アルゴリズム
    optimizer = Optimizer(output_layer, t, name="Optimizer")
    # 評価
    evaluator = Evaluator(output_layer, t, name="Evaluator")

    # トレーニングの実行
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.summary_path, sess.graph)

    # セッションの保存・読み込みを行うオブジェクト
    saver = tf.train.Saver()

    i = 0

    # コマンドライン引数の処理
    if FLAGS.restore_session_path:
        saver.restore(sess, FLAGS.restore_session_path)
        ret = re.search("-(\d+)", FLAGS.restore_session_path)
        i = int(ret.group(1))

    for _ in range(20000):
        i += 1
        # 学習用データのミニバッチを取得する
        batch_x, batch_t = mnist.train.next_batch(50)
        # 学習の実行 ドロップアウト率:0.5
        sess.run(optimizer.train_step, feed_dict={X:batch_x,t:batch_t,keep_prob:0.5})
        if i % 1000 == 0:
            summary_, train_acc, train_loss = sess.run([summary, evaluator.accuracy,optimizer.loss],
                                            feed_dict={X:batch_x,t:batch_t,keep_prob:1.0})
            print "[Train] step: %d, loss: %f, acc: %f" % (i, train_loss, train_acc)
            writer.add_summary(summary_, i)
            # テストデータによるモデルの評価
            # ドロップアウトを適用しないため、keep_prob:1.0
            test_x, test_t = mnist.test.images,mnist.test.labels
            test_acc, test_loss = sess.run([evaluator.accuracy,optimizer.loss], feed_dict={X:test_x,t:test_t,keep_prob:1.0})
            print "[Test] step: %d, loss: %f, acc: %f" % (i, test_loss, test_acc)
            # セッションの保存
            saver.save(sess, FLAGS.summary_path, global_step=i)
    sess.close()
```

実行結果 :

```
$ python mnist_cnn.py --summary_path ./summary_log --session_path ./session_log
[Train] step: 1000, loss: 0.153830, acc: 0.960000
[Test] step: 1000, loss: 0.127999, acc: 0.963500
[Train] step: 2000, loss: 0.071363, acc: 0.960000
[Test] step: 2000, loss: 0.079683, acc: 0.975100
...略...
[Train] step: 19000, loss: 0.000496, acc: 1.000000
[Test] step: 19000, loss: 0.023757, acc: 0.992700
[Train] step: 20000, loss: 0.000214, acc: 1.000000
[Test] step: 20000, loss: 0.024871, acc: 0.992600
```

セッションを読み込みんで、学習を再開する

```
$ python mnist_cnn.py --summary_path ./summary_log --session_path ./session_log --restore_session_path ./summary_log-20000
[Train] step: 21000, loss: 0.000479, acc: 1.000000
[Test] step: 21000, loss: 0.027168, acc: 0.991400
[Train] step: 22000, loss: 0.002611, acc: 1.000000
[Test] step: 22000, loss: 0.028818, acc: 0.993100
...略...
[Train] step: 39000, loss: 0.000082, acc: 1.000000
[Test] step: 39000, loss: 0.030697, acc: 0.992600
[Train] step: 40000, loss: 0.000004, acc: 1.000000
[Test] step: 40000, loss: 0.028255, acc: 0.993100
```
