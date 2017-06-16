# 結合関数

Sample

```python
# coding:utf-8
import numpy as np

# 3x3行列
x = np.arange(0, 9).reshape(3,3)
# 3x3行列
y = np.random.randint(0, 10, (3,3))
print x
print y

# 行列の結合
# 行方向に行列を結合する(デフォルト)
print np.concatenate((x,y), axis=0)
# 以下も同様
# print np.vstack((x,y))
# 列方向に行列を結合する
print np.concatenate((x,y), axis=1)
# 以下も同様
# print np.hstack((x,y))

# 同じ行と列の要素を並べる
print np.dstack((x,y))

# 3x3行列を、1列目、2列目で分割する
x1, x2, x3 = np.split(x,(1,2))
print x1
print x2
print x3
```

実行結果

```
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[9 1 5]
 [3 3 4]
 [0 4 1]]
[[0 1 2]
 [3 4 5]
 [6 7 8]
 [9 1 5]
 [3 3 4]
 [0 4 1]]
[[0 1 2 9 1 5]
 [3 4 5 3 3 4]
 [6 7 8 0 4 1]]
[[[0 9]
  [1 1]
  [2 5]]

 [[3 3]
  [4 3]
  [5 4]]

 [[6 0]
  [7 4]
  [8 1]]]
[[0 1 2]]
[[3 4 5]]
[[6 7 8]]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_concat.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_concat.ipynb)
