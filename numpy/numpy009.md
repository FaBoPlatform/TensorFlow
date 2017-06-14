# ブールインデックス

Sample

```python
# coding:utf-8
import numpy as np

# 標準正規分布による乱数を10コ生成する
x = np.random.randn(10)
print x

# 0以上の場合にはTrueを返す
mask = x >= 0.0
print mask

# Trueとなっている値のみ取得する
print x[mask]
# 以下も同様
print x[x >= 0]
```

出力結果

```shell
[-0.446617   -0.75362278 -1.75269474  0.12279827  1.49845472 -1.02476642
 -0.73013694  0.20197336  0.9706688  -0.35600103]
[False False False  True  True False False  True  True False]
[ 0.12279827  1.49845472  0.20197336  0.9706688 ]
[ 0.12279827  1.49845472  0.20197336  0.9706688 ]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy009.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy009.ipynb)
