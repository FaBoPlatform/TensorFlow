# 行列の演算

## Sample

|演算|説明|
|:-:|:-:|
|+|行列の和|
|-|行列の差|
|np.dot(a, b) または a.dot(b)|行列の積|
|*|行列の**要素**同士の積|
|/|行列の**要素**同士の商|

```python
# coding:utf-8
import numpy as np

# 2x2行列a
a = np.array([[2.0, 1.0], [4.0, 2.0]])
# 2x2行列b
b = np.array([[1.0, 1.0], [6.0, 3.0]])
# 行列の和
print a + b
# 行列の差
print a - b
# 行列の積
print np.dot(a, b)
# 以下も同様
print a.dot(b)
# 行列の要素同士の積
print a * b
# 行列の要素同士の商
print a / b
```

出力結果

```shell
[[  3.   2.]
 [ 10.   5.]]
[[ 1.  0.]
 [-2. -1.]]
[[  8.   5.]
 [ 16.  10.]]
[[  8.   5.]
 [ 16.  10.]]
[[  2.   1.]
 [ 24.   6.]]
[[ 2.          1.        ]
 [ 0.66666667  0.66666667]]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy006.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy006.ipynb)
