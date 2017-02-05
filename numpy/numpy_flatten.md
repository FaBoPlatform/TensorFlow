# 平坦化

Sample

```py
# coding:utf-8
import numpy as np

# 3x3行列
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]
x = np.arange(9).reshape(3,3)
print x

# 行列を平坦化(配列化)する
print x.flatten()
# 以下も同様
# ただし、データのコピーを返さない
print x.ravel()
```

実行結果

```
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[0 1 2 3 4 5 6 7 8]
[0 1 2 3 4 5 6 7 8]
```
