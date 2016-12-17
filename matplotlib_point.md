# 点をうつ

## Sample

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.randn(500,2) + np.array([0,-2])
x2 = np.random.randn(500,2) + np.array([0,2])

plt.figure(1)
plt.plot(x1[:,0], x1[:,1], 'ro', label='Data x1')
plt.plot(x2[:,0], x2[:,1], 'bo', label='Data x2')
plt.show()
```

結果
![](/img/matplotlib_point.png)

