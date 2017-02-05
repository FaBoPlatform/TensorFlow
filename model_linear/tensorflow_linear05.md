# 線形回帰 課題

## 課題1

![](/img/kadai01.png)

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

b_train = -1
w_train = 0.7
x_train = np.random.random((1, 100))
y_train = x_train * w_train + b_train + 0.1 * np.random.randn(1, 100)

plt.figure(1)
plt.plot(x_train, y_train, 'ro', label='Data')
plt.show()
```

正規分布で散らばった値の線形回帰のb,wを求めるプログラムをTensorFlowで作成せよ　

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/kadai01.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/kadai01.ipynb)

## 課題2

![](/img/kadai02.png)

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

x_train = np.linspace(0, 100, 100)
y_train = x_train + 10 * np.sin(x_train/10)

plt.figure(1)
plt.plot(x_train, y_train, 'ro', label='Data')
plt.show()
```

非線形に散らばった値の線形回帰のb,wを求めるプログラムをTensorFlowで作成せよ　

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/kadai02.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/kadai02.ipynb)

