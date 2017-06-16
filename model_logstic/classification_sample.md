# 乱数からデータセットを作り、クラス分類を行う
---

## データセット作成
`np.random.rand(N,3)`から乱数を生成し、特徴データとして扱う。  
今回は
* 0.0~3.0の間の数を classC (例: [1.05,2.21,1.26])
* 3.0~6.0の間の数を classB (例: [4.25,4.76,5.19])
* 6.0~10.0の間の数を classA  (例: [6.24,8,56,9,92])

とし、3クラス分類を行うための乱数を生成する。

Sampleコード   
```python
import numpy as np
import csv

# ファイルオープン
f = open('class_train.csv', 'a')
writer = csv.writer(f, lineterminator='\n')

x = np.add((10.0-6.0)*np.random.rand(100,3),6.0)
x = x.astype(np.float32)

for i in range (0,x.shape[0]):
  # データをリストに保持
  csvlist = []
  for n in x[i]:
    csvlist.append(n)
  csvlist.append("classA")
  writer.writerow(csvlist)

x = np.add((6.0-3.0)*np.random.rand(100,3),3.0)
x = x.astype(np.float32)

for i in range (0,x.shape[0]):
  # データをリストに保持
  csvlist = []
  for n in x[i]:
    csvlist.append(n)
  csvlist.append("classB")
  writer.writerow(csvlist)

x = np.add((3.0-0.0)*np.random.rand(100,3),0.0)
x = x.astype(np.float32)

for i in range (0,x.shape[0]):
  # データをリストに保持
  csvlist = []
  for n in x[i]:
    csvlist.append(n)
  csvlist.append("classC")
  writer.writerow(csvlist)

# ファイルクローズ
f.close()
```

上記のSampleコードを実行したカレントディレクトリに`class_train.csv`が作成されていることを確認する。  
また以下のようなものが各クラス100個作成される。  
![](/img/classification_sample001.png)  
同様にテスト用のcsvも作成しておくこと。(データ数は任意)

## トレーニング
作成したcsvから学習を行う。  
Sampleコード  
```python
import numpy as np
import tensorflow as tf

### データの準備
# データセットの読み込み
train_dataset = np.genfromtxt("./class_train.csv", delimiter=',', dtype=[float, float, float, "S32"])
test_dataset = np.genfromtxt("./class_test.csv", delimiter=',', dtype=[float, float, float, "S32"])

#データをシャッフル
np.random.shuffle(train_dataset)
np.random.shuffle(test_dataset)

def get_labels(dataset):
    """ラベル(正解データ)を1ofKベクトルに変換する"""
    raw_labels = [item[3] for item in dataset]
    labels = []
    for l in raw_labels:
        if l == "classA":
            labels.append([1.0,0.0,0.0])
        elif l == "classB":
            labels.append([0.0,1.0,0.0])
        elif l == "classC":
            labels.append([0.0,0.0,1.0])
    return np.array(labels)

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[:3] for item in dataset]
    return np.array(raw_data)

# 学習ラベル
t_train = get_labels(train_dataset)
# 学習データ
x_train = get_data(train_dataset)
# テストデータ
t_test = get_labels(test_dataset)
x_test = get_data(test_dataset)

# ラベルを格納するPlaceholder
t = tf.placeholder(tf.float32, shape=(None,3))
# データを格納するPlaceholder
X = tf.placeholder(tf.float32, shape=(None,3))

def single_layer(X):
    """隠れ層"""
    node_num = 30
    w = tf.Variable(tf.truncated_normal([3,node_num]))
    b = tf.Variable(tf.zeros([node_num]))
    f = tf.matmul(X, w) + b
    layer = tf.nn.relu(f)
    return layer

def output_layer(layer):
    """出力層"""
    node_num = 30
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
# 勾配降下法 学習率0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)
# モデルの予測と正解が一致しているか調べる
correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
# モデルの精度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
for _ in range(2000):
    i += 1
    # トレーニング
    sess.run(train_step, feed_dict={X:x_train,t:t_train})
    # 200ステップごとに精度を出力
    if i % 200 == 0:
        # コストと精度を出力
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X:x_train,t:t_train})
        # テスト用データを使って評価
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X:x_test,t:t_test})
        print "Step: %d" % i
        print "[Train] cost: %f, acc: %f" % (train_loss, train_acc)
        print "[Test] cost: %f, acc: %f" % (test_loss, test_acc)
```

トレーニングの途中経過  
![](/img/classification_sample002.png)   

## モデルのテスト  
作成したモデルと未知のデータで予測を行なってみる。  
Sampleコード  
```python
test = np.array([8.3,8.1,8.2])
test = np.array([test])
ans = sess.run(p,feed_dict={X:test})
print(ans)
tmp = np.argmax(ans,axis=1)
if(tmp == 0):
  print("classA")
elif(tmp == 1):
  print("classB")
elif(tmp == 2):
  print("classC")

test = np.array([0.23,2.11,1.15])
test = np.array([test])
ans = sess.run(p,feed_dict={X:test})
print(ans)
tmp = np.argmax(ans,axis=1)
if(tmp == 0):
  print("classA")
elif(tmp == 1):
  print("classB")
elif(tmp == 2):
  print("classC")
```
結果  
![](/img/classification_sample003.png)   
うまく予測ができた。  

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/training_Classification.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/training_Classification.ipynb)
