# さまざまな分布に従う乱数

|関数名|説明|
|:-:|:-:|
|`np.random.binomial(n,p,size)`|二項分布|
|`np.random.normal(loc,scale,size)`|正規分布|
|`np.random.beta(a,b,size)`|ベータ分布|
|`np.random.chisquare(df,size)`|カイ二乗分布|
|`np.random.gamma(shape,scale,size)`|ガンマ関数|
|`np.random.uniform(low,high,size)`|区間0<=x<1の一様分布|

Sample

```python
# coding:utf-8
import numpy as np

# 二項分布に従う乱数
binomial = np.random.binomial(10, 0.5, (2,2))
print binomial
# 正規分布に従う乱数
normal = np.random.normal(0, 1, (2,2))
print normal
# ベータ分布に従う乱数
beta = np.random.beta(1, 1, (2,2))
print beta
# カイ二乗分布に従う乱数
chisquare = np.random.chisquare(1, (2,2))
print chisquare
# ガンマ分布
gamma = np.random.gamma(2, 1, (2,2))
print gamma
# [0,1)の一様分布
uniform = np.random.uniform(0, 1, (2,2))
print uniform
```

実行結果

```
[[5 4]
 [6 2]]
[[ 1.09813924  0.1587407 ]
 [-0.49877663  0.36559575]]
[[ 0.67614291  0.14561996]
 [ 0.71546136  0.7953883 ]]
[[ 0.67752021  4.52508705]
 [ 0.38940294  0.05195434]]
[[ 1.16165311  1.95626104]
 [ 3.65589006  4.50888572]]
[[ 0.43489813  0.25007418]
 [ 0.42769298  0.19222238]]
```

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_random.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/numpy/numpy_random.ipynb)
