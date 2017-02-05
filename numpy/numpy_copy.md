# Numpyオブジェクトのコピー

Sample

```python
# coding:utf-8
import numpy as np

# 乱数のシードを設定する
np.random.seed(20200724)

# 標準正規分布に従う乱数を要素に持つ4x4行列
x = np.random.randn(2, 2)
print x

# Numpyオブジェクトのコピー
y = x.copy()
print y
```

実行結果

```
[[ 0.061722   -0.57419489]
 [-0.22935131  0.96591975]]
[[ 0.061722   -0.57419489]
 [-0.22935131  0.96591975]]
```