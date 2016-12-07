# 次元の増加

Sample

```python
# coding:utf-8
import numpy as np

x = np.arange(4)
print x
# 次元数を増やす
print x[np.newaxis,:]
# 縦方向に次元数を増やす
print x[:,np.newaxis]
```

実行結果

```
[0 1 2 3]
[[0 1 2 3]]
[[0]
 [1]
 [2]
 [3]]
```
