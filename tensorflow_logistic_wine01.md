# ワイン評価

> curl http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.c
sv > wine.cvs

データ・セットのフォーマットは、下記の通り。

![](/img/wine01.png)

このデータをdataとlabelに分離する。

![](/img/wine02.png)

Labelの1-10の評価は下記のようにベクトル化する

![](/img/wine03.png)

```python
# coding:utf-8                                                                                                                
import numpy as np
import tensorflow as tf
import pandas as pd

# データセットを読み込む
# なお欠損値は0とした 
dataset = np.genfromtxt("./wine.csv", delimiter=';', dtype=np.float32, filling_values=(0))

# 重複したデータを省く
_,index = np.unique(dataset[:,0], return_index=True)
dataset = dataset[index]

# Wineのデータ
datas = dataset[:,0:11]

# Wineのラベル(品質)
labels = dataset[:,11]
N = len(labels)
vector_labels = np.zeros((N,10))
for i in xrange(N):
    vector_labels[i][int(labels[i])] = 1.0

# データを7:1で分割する
train_data_size = len(dataset) - len(dataset) // 8
test_data_size = len(dataset) // 8

# 訓練用データ
train_datas = datas[:train_data_size]
train_labels = vector_labels[:train_data_size].reshape(train_data_size, 1)

# テスト用データ
test_datas = datas[train_data_size:]
test_labels = vector_labels[train_data_size:].reshape(test_data_size, 1)

# データを1件抽出し、表示する
print train_datas[0]
print train_labels[0]

```

出力結果
```shell
[  4.59999990e+00   5.19999981e-01   1.50000006e-01   2.09999990e+00
   5.40000014e-02   8.00000000e+00   6.50000000e+01   9.93399978e-01
   3.90000010e+00   5.60000002e-01   1.31000004e+01]
[ 0,0,0,1.0,0,0,0,0,0,0]
```