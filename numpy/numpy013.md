# numpyオブジェクトの型・次元数・サイズ

## Sample

```python
# coding:utf-8
import numpy as np

# 3x3行列
x = np.random.randn(3, 3)
print x
# 型
print x.dtype
# 次元数
print x.shape
# サイズ
print x.size

# 2x2x2多次元配列
y = np.arange(8).reshape(2,2,2)
print y
# 型
print y.dtype
# 次元数
print y.shape
# サイズ
print y.size
```

出力結果

```shell
[[-1.70284321  1.3380814  -0.51146346]
 [ 1.6092976  -0.65849397 -1.46087059]
 [ 0.9852956   0.15465021 -0.18471899]]
float64
(3, 3)
9
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
int64
(2, 2, 2)
8
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy013.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy013.ipynb)
