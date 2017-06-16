# 平均・分散・標準偏差

## Sample

```python
# coding:utf-8
import numpy as np

# 3x3行列
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]
x = np.arange(9).reshape(3,3)
print x

# 平均
print x.mean()
# 分散
print x.var()
# 標準偏差
print x.std()

# 列単位での平均・分散・標準偏差
print x.mean(0)
print x.var(0)
print x.std(0)

# 行単位での平均・分散・標準偏差
print x.mean(1)
print x.var(1)
print x.std(1)
```

出力結果

```shell
[[0 1 2]
 [3 4 5]
 [6 7 8]]
4.0
6.66666666667
2.58198889747
[ 3.  4.  5.]
[ 6.  6.  6.]
[ 2.44948974  2.44948974  2.44948974]
[ 1.  4.  7.]
[ 0.66666667  0.66666667  0.66666667]
[ 0.81649658  0.81649658  0.81649658]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy014.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy014.ipynb)
