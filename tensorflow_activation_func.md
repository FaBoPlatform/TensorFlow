# 活性化関数・出力層の関数

ニューロンをどのように活性化させるかを決める活性化関数および出力層の関数のサンプル

かつてはシグモイド関数が一般的に活性化関数として使われていたが、現在はReLU関数がよく用いられている。

出力層に用いる関数は以下の表のように目的に応じて選択する。

|問題の種別|出力層の活性化関数|
|:-:|:-:|
|回帰|恒等写像|
|二値分類|ロジスティクス関数(シグモイド関数)|
|多クラス分類|ソフトマックス関数|

> 岡谷貴之,"深層学習" p.15 一部修正(誤差関数は省略)

Sample

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -1.0から1.0まで値を0.01間隔で取得
x_data = np.arange(-30.0, 30.0, 0.1)
x = tf.constant(x_data , tf.float32)
#y_data = np.arange(-100.0, 100.0, 0.1).reshape(1,2000)
#print y_data

# シグモイド関数
sigmoid = tf.sigmoid(x)
# ハイパボリックタンジェント(双曲線関数)
tanh = tf.tanh(x)
# ReLU関数
relu = tf.nn.relu(x)

# ソフトマックス関数
# f1:y = -0.5x
# f2:y = 0.25x
# f3:y = 0.5x-5.0
# 600x1行列
xx = tf.placeholder(tf.float32, shape=(600, 1))
# 1x3行列
w = tf.constant([-0.5,0.25,0.5] , tf.float32, shape=(1,3))
# 600x3行列
b = tf.constant(np.array([[0.0,0.0,-5]] * 600), tf.float32, shape=(600,3))
# 行列の積
f = tf.matmul(xx,w)+b
softmax = tf.nn.softmax(f)

with tf.Session() as sess:
    # シグモイド関数
    sigmoid_y = sess.run(sigmoid)
    # ハイパボリックタンジェント
    tanh_y = sess.run(tanh)
    # ReLU関数
    relu_y = sess.run(relu)
    # ソフトマックス関数
    softmax_y = sess.run(softmax, feed_dict={xx:x_data.reshape(600,1)})

    # プロット(グラフを4分割)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # シグモイド関数
    ax1.set_ylim([-0.1,1.1])
    ax1.plot(x_data, sigmoid_y, label="sigmoid")
    ax1.legend(loc="upper left", fontsize=8)
    # ハイパボリックタンジェント
    ax2.set_ylim([-1.1,1.1])
    ax2.plot(x_data, tanh_y, label="tanh")
    ax2.legend(loc="upper left", fontsize=8)
    # ReLU関数
    ax3.set_ylim([-0.1,30])
    ax3.plot(x_data, relu_y, label="Relu")
    ax3.legend(loc="upper left", fontsize=8)
    # ソフトマックス関数
    ax4.set_ylim([-0.05,1.1])
    ax4.plot(x_data, softmax_y[:,0], label="softmax(f1)")
    ax4.plot(x_data, softmax_y[:,1], label="softmax(f2)")
    ax4.plot(x_data, softmax_y[:,2], label="softmax(f3)")
    ax4.legend(loc="upper left", fontsize=8)
    # グラフを表示する
    plt.show()
```

実行結果 : 

![](/img/tensorflow_activation_func.png)

## 参考

* https://www.tensorflow.org/versions/master/api_docs/python/nn.html#activation-functions
* 活性化関数の解説
    * https://ja.wikipedia.org/wiki/%E6%B4%BB%E6%80%A7%E5%8C%96%E9%96%A2%E6%95%B0
