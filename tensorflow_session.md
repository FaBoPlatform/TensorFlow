# セッションの保存・読み込み

セッション内の`Variables`を保存・読み込む。モデルの学習時にパラメタの値を保存しておきたい場合に利用する。

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# 3x3行列の乱数生成オペレーション
rand_op = tf.random_normal(shape=(3,3))
# 3x3行列のVariable このノードが保存される
x = tf.Variable(tf.zeros(shape=(3,3)))
# xに3x3の乱数行列を割り当てるオペレーション
update_x = tf.assign(x, rand_op)

# セッションの保存・読み込みを行うオブジェクト
saver = tf.train.Saver()

# 保存用のセッション
# rand_opの実行ごとにxノードには違う乱数が格納される
# そのときのセッションが保存される
sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())
for i in range(0, 3):
    # rand_opを実行して、3x3行列を生成し、xに割り当てる
    sess1.run(update_x)
    # xの値を表示する
    print sess1.run(x)
    # セッション情報を保存する
    saver.save(sess1, "./rand", global_step=i)

# セッションの読み込み
sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())
# 最後のセッションを読み込む
saver.restore(sess2, "./rand-2")
print sess2.run(x)
```

以下のようなファイルが生成させる

```shell
$ ls rand*
rand-0.data-00000-of-00001 rand-1.data-00000-of-00001 rand-2.data-00000-of-00001
rand-0.index               rand-1.index               rand-2.index
rand-0.meta                rand-1.meta                rand-2.meta
```

出力結果

```shell
[[-0.95573789 -0.3942928   0.62031686]
 [-0.36327139 -1.11739874  0.29342058]
 [-0.60707402  0.31450525  1.39899957]]
[[ 0.73793924  0.20724617  0.68963534]
 [ 0.77061969 -0.31082281  0.73349142]
 [ 0.36134794 -0.28316727 -1.11089706]]
[[ 0.26471373  1.14373958  2.34996057]
 [ 1.51682055 -0.50154948  0.65435147]
 [-1.06940079 -0.65639728 -1.31213081]]
[[ 0.26471373  1.14373958  2.34996057]
 [ 1.51682055 -0.50154948  0.65435147]
 [-1.06940079 -0.65639728 -1.31213081]]
```

## ノート

セッションの読み込み時にファイルパスを"ファイル名"とすると失敗する可能性あり。"./ファイル名"としたところ成功した。

実行環境:

* Python 2.7.12 :: Anaconda 4.1.1 (x86_64)
* TensorFlow 0.12.0-rc0

類似の不具合:

* https://github.com/tensorflow/tensorflow/issues/571

## 参考


