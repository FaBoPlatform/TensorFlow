# ロジスティック回帰の準備

breast-cancer-wisconsinデータセットを使い、TensorFlowによるロジスティック回帰を行う。breast-cancer-wisconsinデータセットはがん細胞の情報とがんの悪性か良性かをまとめたデータセットである。

データセットのダウンロード

```
$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
```

Sample

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

# データセットを読み込む
# なお欠損値は0とした
dataset = np.genfromtxt("./breast-cancer-wisconsin.data", delimiter=',', dtype=np.uint32, filling_values=(0))

# 重複したデータを省く
_,index = np.unique(dataset[:,0], return_index=True)
dataset = dataset[index]
# 2が良性 => 0に置き換える
dataset[:,10][np.where(dataset[:,10] == 2)] = 0
# 4が悪性 => 1に置き換える
dataset[:,10][np.where(dataset[:,10] == 4)] = 1

# 患者のデータ
data = dataset[:,1:10]

# ラベル(正解データ)
labels = dataset[:,10]

# データを7:1で分割する
# 全データ数 : 699
# 訓練用データ数 : 612
# テスト用データ数 : 87
train_data_size = len(dataset) - len(dataset) // 8
test_data_size = len(dataset) // 8

# 訓練用データ
train_data = data[:train_data_size]
train_labels = labels[:train_data_size].reshape(train_data_size, 1)

# テスト用データ
test_data =data[train_data_size:]
test_labels = labels[train_data_size:].reshape(test_data_size, 1)

# データを1件抽出し、表示する
print train_data[1]
print train_labels[1]
```

実行結果

```
[ 9  1  2  6  4 10  7  7  2]
[1]
```

## 参考

* [Breast Cancer Wisconsin (Prognostic) Data Set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Prognostic))
