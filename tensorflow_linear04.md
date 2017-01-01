# 線形回帰 解析

## GradientDescentOptimizerのlearning_rateを変えてみる

```python

	# 勾配降下法
	learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    ...

    # トレーニング回数
    training_step = 5000
    validation_step = 10
```

![](/img/linear_test_w001.png) ![](/img/linear_test_b001.png) ![](/img/linear_test_loss001.png)

## AdamOptimizer

```python

	# Optimizer
	train_op = tf.train.AdamOptimizer().minimize(loss)
    ...

    # トレーニング回数
    training_step = 10000
    validation_step = 10
```

![](/img/linear_test_w002.png) ![](/img/linear_test_b002.png) ![](/img/linear_test_loss002.png)

## AdadeltaOptimizer

```python

	# Optimizer
	learning_rate = 0.5
    train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    ...
     # トレーニング回数
    training_step = 10000
    validation_step = 10
```

![](/img/linear_test_w003.png) ![](/img/linear_test_b003.png) ![](/img/linear_test_loss003.png)

## AdagradOptimizer

```python

	# Optimizer
	learning_rate = 0.025
    train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    ...
     # トレーニング回数
    training_step = 10000
    validation_step = 10
```

![](/img/linear_test_w004.png) ![](/img/linear_test_b004.png) ![](/img/linear_test_loss004.png)

## MomentumOptimizer

```python

	# Optimizer       
    learning_rate = 0.01
    momentum_rate = 0.01
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum_rate).minimize(loss)
    ...
     # トレーニング回数
    training_step = 10000
    validation_step = 10
```

![](/img/linear_test_w005.png) ![](/img/linear_test_b005.png) ![](/img/linear_test_loss005.png)


