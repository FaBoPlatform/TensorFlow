# 転置行列

Sample

```python
# coding:utf-8
import numpy as np

# 3x3行列
x = np.arange(9).reshape(3,3)
print x
# 転置行列
# 例 1行2列目の要素は2行1列目の要素と入れ替わる
print x.T
```

出力結果

```shell
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[0 3 6]
 [1 4 7]
 [2 5 8]]
```
