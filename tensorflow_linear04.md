# 線形回帰 課題

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


