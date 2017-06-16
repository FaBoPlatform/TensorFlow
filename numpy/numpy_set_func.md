# 集合関数

Numpyオブジェクトに対する集合関数のサンプル

|||
|:-:|:-:|
|`np.unique(x)`|重複した値を省く|
|`np.intersect1d(x,y)`|積集合|
|`np.union1d(x,y)`|和集合|
|`np.setdiff1d(x,y)`|差集合|
|`np.setxor1d(x,y)`|排他的論理和|
|`np.in1d(x,y)`|包含|

Sample

```python
# coding:utf-8
import numpy as np

# 乱数のシードを設定する
np.random.seed(20200724)

# 範囲0〜4の乱数を10個生成する
x = np.random.randint(0, 4, 10)
print x

# 重複した値を取り除く
print np.unique(x)

# 集合演算
# [0, 1, 2, 3, 4]
a = np.arange(5)
# [3, 4, 5, 6, 7]
b = np.arange(3, 8)

print a
print b
# 積集合 a ∩ b
print np.intersect1d(a, b)
# 和集合 a ∪ b
print np.union1d(a, b)
# 差集合 a - b
print np.setdiff1d(a, b)
# 排他的論理和 a xor b
print np.setxor1d(a, b)
# 配列bの各要素が、配列aの中に含まれているか調べる
print np.in1d(a, b)
```

実行結果

```
[1 0 3 1 3 1 2 0 1 3]
[0 1 2 3]
[0 1 2 3 4]
[3 4 5 6 7]
[3 4]
[0 1 2 3 4 5 6 7]
[0 1 2]
[0 1 2 5 6 7]
[False False False  True  True]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_repeat.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_set_func.ipynb)
