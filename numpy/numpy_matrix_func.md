# 行列関数

行列の計算を行う数学関数のサンプル

|関数名|説明|
|:-:|:-:|
|`np.linalg.norm(x)`|ノルム|
|`np.diag(x)`|ベクトルを対角行列にする|
|`np.linalg.det(x)`|行列式|
|`np.linalg.inv(x)`|逆行列|
|`np.dot(x)`|行列の内積|
|`np.trace(x)`|対角和|
|`np.linalg.eig(x)`|固有行列|
|`np.eye(x)`|単位行列|
|`np.linalg.solve(x, y)`|連立方程式の解|

Sample

```python
# coding:utf-8
import numpy as np

# [ 1.  2.  3.]
x = np.arange(1.0, 4.0)
print x

# ベクトルのノルム(距離)
print np.linalg.norm(x)

# ベクトルを対角行列にする
diag_x = np.diag(x)

# 行列式
print np.linalg.det(diag_x)

# 逆行列
print np.linalg.inv(diag_x)

# 行列の内積
# 3x3単位行列
e = np.eye(3)
print np.dot(diag_x, e)

# 対角和
print np.trace(diag_x)

# 固有値、固有ベクトル
print np.linalg.eig(diag_x)

# 連立方程式の解
# 2x+y+z = 15
# 4x+6y+3z = 41
# 8x+8y+9z = 83
# 解 : x=5,y=2,z=3
# 薩摩順吉, 四ツ谷晶二, "キーポイント線形代数" p.2より
a = np.array([[2,1,1],[4,6,3],[8,8,9]])
b = np.array([[15],[41],[83]])
print np.linalg.solve(a, b)
```

実行結果

```
[ 1.  2.  3.]
3.74165738677
6.0
[[ 1.          0.          0.        ]
 [ 0.          0.5         0.        ]
 [ 0.          0.          0.33333333]]
[[ 1.  0.  0.]
 [ 0.  2.  0.]
 [ 0.  0.  3.]]
6.0
(array([ 1.,  2.,  3.]), array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]]))
[[ 5.]
 [ 2.]
 [ 3.]]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_matrix_func.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_matrix_func.ipynb)
