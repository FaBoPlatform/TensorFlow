# 配列の繰り返し

Sample

```python
# coding:utf-8
import numpy as np

# 配列の各要素を4つずつ並べる
x = np.array([3,6,9]).repeat(4)
print x

# 配列の内容を3つ並べる
y = np.tile([7,2,4], 3)
print y

# 配列の内容を2x2で並べる
z = np.tile([7,2,4], (2,2))
print z
```

実行結果

```
[3 3 3 3 6 6 6 6 9 9 9 9]
[7 2 4 7 2 4 7 2 4]
[[7 2 4 7 2 4]
 [7 2 4 7 2 4]]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_repeat.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_repeat.ipynb)
