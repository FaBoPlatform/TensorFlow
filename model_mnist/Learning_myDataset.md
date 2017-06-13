# 任意のデータセットでMNISTのような画像認識を行う　その2
---
## トレーニング  
用意したデータセットで学習を行う。  
ここでは畳み込みではなく、全結合のみで学習を行った。   
読み込みのSampleコードを`load_dataset.py`で保存し、使用している。  
Sampleコード  
```python
import numpy as np
import tensorflow as tf
from load_dataset import load_dataset
import matplotlib.pyplot as plt

x_train,t_train = load_dataset('./train_dataset',convert_type='RGB',flatten=True,normalize=True,one_hot_label=True)
x_test,t_test = load_dataset('./test_dataset',convert_type='RGB',flatten=True,normalize=True,one_hot_label=True)

#データのシャッフル
def list_shuffle(datas,labels):
    index_list = np.arange(0,datas.shape[0])
    np.random.shuffle(index_list)
    x_data = datas[index_list]
    t_data = labels[index_list]
    return x_data,t_data

x_train_shuffle,t_train_shuffle = list_shuffle(x_train,t_train)
x_test_shuffle,t_test_shuffle = list_shuffle(x_test,t_test)

# 学習
# 入力層
# 任意の用意した画像サイズ,分類のクラス数を指定すること
x = tf.placeholder(tf.float32, shape=[None, 64*64*3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
W = tf.Variable(tf.zeros([64*64*3,2]))
b = tf.Variable(tf.zeros([2]))

#  出力層
y = tf.matmul(x,W) + b
p = tf.nn.softmax(y)
# 損失関数
cross_entropy = y_ * tf.log(p)
loss = -tf.reduce_mean(cross_entropy)
# 勾配
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    sess.run(train_step,feed_dict={x: x_train_shuffle, y_: t_train_shuffle})
    if i % 100 == 0:
        train_acc, train_loss = sess.run([accuracy,loss], feed_dict={x: x_train_shuffle, y_: t_train_shuffle})
        test_acc = sess.run(accuracy, feed_dict={x: x_test_shuffle, y_: t_test_shuffle})
        print "[Train] step: %d, loss: %f, acc: %f, [Test] acc : %f" % (i, train_loss, train_acc,test_acc)
```

作成したモデルで新しいデータを入力してみる  
ここではクラス分類を道路標識の止まれと制限速度の標識の2クラス分類。  
Sample
```python
from PIL import Image
im = Image.open("test.jpg", "r")
plt.imshow(np.array(im))
img = np.frombuffer(np.array(Image.open('test.jpg').convert('RGB')),dtype=np.uint8)
img = img.astype(np.float32)
# 正規化
img /= 255.0
predict_img = np.array([img])
ans = np.argmax(sess.run(p,feed_dict={x:predict_img}))
if(ans == 0):
  print('Mark is Stop')
elif(ans == 1):
  print('Mark is limitspeed')
```

結果  
![](/img/Learning_myDataset001.png)  
