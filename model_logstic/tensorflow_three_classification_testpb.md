# 3クラス分類 PBファイルテスト編

irisデータセットを使った3クラス分類の学習モデルのpbファイルの精度の確認を行う

サンプルコード :
```python
# coding:utf-8
# TensorFlow r1.0.0
# Python 2.7.6
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#pbファイルを読み込む
with tf.gfile.FastGFile("./tmp/iris-practice/iris-graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#学習モデルの精度テスト    
sess = tf.Session()

# Iris-virginicaのサンプルデータ
input_data = [[7.7,3.8,6.7,2.2]]
#グラフにつけたname="output"とname="input"に入力データと出力結果を代入する
predictions = sess.run('output:0',{'input:0':np.array(input_data)})
print predictions

Index = np.argmax(predictions)
if(Index == 0):
    print "answer : Iris-setosa"
elif(Index == 1):
    print "answer : Iris-versicolor"
elif(Index == 2):
    print "answer : Iris-virginica"
```

実行結果 :

```
[[ 0.11481573  0.42691165  0.45827267]]
answer : Iris-virginica
```

```python
Index = np.argmax(predictions)
if(Index == 0):
    print "answer : Iris-setosa"
elif(Index == 1):
    print "answer : Iris-versicolor"
elif(Index == 2):
    print "answer : Iris-virginica"
```
predictionsには実行結果の通り、どのクラスが尤もらしいかの値が格納されている。
その値が一番大きい配列のインデックスが入力データの答えとなる
