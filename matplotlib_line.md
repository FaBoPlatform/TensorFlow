# 線を引く

## Sample

```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.randn(500,2) + np.array([0,-2])
x2 = x1 * 0.3 - 1
plt.figure(1)
plt.plot(x1[:,0], x2[:,0], 'k-', label='Data x1')
plt.show()
```

結果
![](/img/matplotlib_line.png)

