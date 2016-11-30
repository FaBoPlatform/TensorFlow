
# Matplotlib

Matplotlibをインストール

```shell
$ pip install matplotlib
```

```shell
vi ~/.matplotlib/matplotlibrc
```

```txt
backend : TkAgg
```

# Sample

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(10,1)
Y = np.random.randn(10,1)

print X
print Y

plt.scatter(X[:,0],Y[:,0], s=100, alpha=0.5)
plt.show()
```



