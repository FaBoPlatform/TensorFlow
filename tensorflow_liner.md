# 線形回帰

線形回帰のサンプル

## Sample

```python
# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 乱数シード
np.random.seed(1)

# トレーニングデータの作成(1x100)
train_X = np.random.random((1, 100))
train_Y = (train_X * 0.7 -1) + 0.1 * np.random.randn(1, 100)

# トレーニング回数
training_step = 100
validation_step = 10

# 変数の定義
trX = tf.placeholder(tf.float32, name = "input")
trY = tf.placeholder(tf.float32, name = "output")
W = tf.Variable(np.random.randn(), name = "weight")
b = tf.Variable(np.random.randn(), name = "bias")

# activation = XW + b
activation = tf.add(tf.mul(trX, W), b)

# Loss Function(損失関数)
# 差が+の時、-の時があるので、2乗して、その平均を出す
cost = tf.reduce_mean(tf.pow(activation - trY, 2))
#cost = tf.reduce_mean(tf.square(activation - Y))

# 勾配降下法
learning_rate = 0.01
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初期化
init_op = tf.initialize_all_variables()

# Session
sess = tf.Session()
sess.run(init_op)

# トレーニング
for step in range(training_step):
	for(x,y) in zip(train_X[0,:], train_Y[0,:]):
		sess.run(train_op, feed_dict={trX:x, trY:y})

	# 途中経過表示
	if step % validation_step == 0:
		cost_val = sess.run(cost, feed_dict={trX:train_X, trY:train_Y})
		weight = sess.run(W)
		bias = sess.run(b)
		print "Step %i, cost %f, weight %f, bias %f" % (step, cost_val, weight, bias) 

# 結果をグラフに出す
result_W = sess.run(W)
result_b = sess.run(b)
linear_Y = train_X * result_W + result_b 

plt.figure(2)
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.plot(train_X[0, :], linear_Y[0, :], 'r-', label='Linner Line')
plt.axis('equal')
plt.legend(loc='lower right')
plt.show()
```

実行結果

```
Step 0, cost 0.052620, weight 0.002306, bias -0.574152
Step 10, cost 0.010669, weight 0.500573, bias -0.879639
Step 20, cost 0.008196, weight 0.627029, bias -0.949147
Step 30, cost 0.008032, weight 0.658357, bias -0.966367
Step 40, cost 0.008019, weight 0.666117, bias -0.970632
Step 50, cost 0.008017, weight 0.668040, bias -0.971689
Step 60, cost 0.008017, weight 0.668516, bias -0.971951
Step 70, cost 0.008017, weight 0.668635, bias -0.972016
Step 80, cost 0.008017, weight 0.668663, bias -0.972031
Step 90, cost 0.008017, weight 0.668670, bias -0.972035
```

![]("/img/tesorflow_linear.png")
