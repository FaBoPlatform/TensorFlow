# 3クラス分類 準備編

irisデータセットを使った3クラス分類のサンプルコード

irisデータセットはアヤメの特徴とそのアヤメの種類をまとめたデータセットである。

irisデータセットのダウンロード

```
$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data
```

サンプルコード :

```python
# coding:utf-8
"""
irisデータセットを使った3クラス分類
"""
import numpy as np
import tensorflow as tf

# データセットの読み込み
# 各列の値の型を指定する必要がある
dataset = np.genfromtxt("./bezdekIris.data", delimiter=',', dtype=[float, float, float, float, "S32"])
# データセットの順序をランダムに並べ替える
np.random.shuffle(dataset)

def get_labels(dataset):
    """ラベル(正解データ)を1ofKベクトルに変換する"""
    raw_labels = [item[4] for item in dataset]
    labels = []
    for l in raw_labels:
        if l == "Iris-setosa":
            labels.append([1,0,0])
        elif l == "Iris-versicolor":
            labels.append([0,1,0])
        elif l == "Iris-virginica":
            labels.append([0,0,1])
    return np.array(labels)

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[:4] for item in dataset]
    return np.array(raw_data)

# ラベル
labels = get_labels(dataset)
# データ
data = get_data(dataset)

# irisデータセットの形式
print labels.shape
print data.shape

# 訓練データとテストデータに分割する
# 訓練用データ
train_labels = labels[:120]
train_data = data[:120]
print train_labels.shape
print train_data.shape

# テスト用データ
test_labels = labels[120:]
test_data = data[120:]
print test_labels.shape
print test_data.shape
```

実行結果 :

```
(150, 3)
(150, 4)
(120, 3)
(120, 4)
(30, 3)
(30, 4)
```

## 参考

* [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/Iris)
