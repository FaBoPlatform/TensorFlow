# 線形回帰 課題

## 課題1

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

b_train = -1
W_train = 0.7
X_train = np.random.random((1, 100))
y_train = X_train * W_train + b_train + 0.1 * np.random.randn(1, 100)

plt.figure(1)
plt.plot(X_train, y_train, 'ro', label='Data')
plt.show()
```

![](/img/linear_kadai001.png)

正規分布で散らばった値の線形回帰のb,Wを求めるプログラムをTensorFlowで作成せよ　

## 課題2

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

X_train = np.linspace(0, 100, 100)
y_train = X_train + 10 * np.sin(X_train/10)

plt.figure(1)
plt.plot(X_train, y_train, 'ro', label='Data')
plt.show()
```

![](/img/linear_kadai002.png)

非線形に散らばった値の線形回帰のb,Wを求めるプログラムをTensorFlowで作成せよ　
