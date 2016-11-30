
# Numpyのインストール

NumpyとScipyをインストール

```shell
$ pip install numpy
$ pip install scipy
```

# Sample1

```shell
import numpy as np

X = np.random.randn(10, 2)

print X
```

出力結果
```shell
[[ 1.29809056  1.81463907]
 [ 0.73759474 -0.5688174 ]
 [-1.45300168  0.2555659 ]
 [-0.18510103 -0.18096903]
 [ 0.37652474  0.46395745]
 [-0.45630759 -0.41358571]
 [-1.55313916 -0.09374661]
 [ 1.27759706  2.21715869]
 [ 0.08562266  0.67993018]
 [ 0.14558376  0.37880808]]
```

# Sample2

```python
import numpy as np
array = np.array([1,5,10,4,11,22,21,11,10,1])
print array

array = array.reshape([5,2])

print array
```

# Sample3

```python
import numpy as np

array = np.zeros([3,2])
print array
```

# Sample4

```python
import numpy as np

matrix = np.random.randn(20,3)
print matrix
```

# Sample5

```python
import numpy as np

array = np.array([0]*100)

print array
```
