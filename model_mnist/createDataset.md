# 任意のデータセットでMNISTのような画像認識を行う　その1
---
## データセット作成
学習用とテスト用の画像をネット上から集めてくる、もしくは配布されているデータセットをダウンロードする。  

### 特徴の切り抜き
MNISTのように画像をリサイズ、または特徴を切り抜いてリサイズを行う。  
![](/img/createDataset001.png)  

今回特徴の切り抜きとリサイズには[CattingImage](https://github.com/yamika/Catting_Image.git)を使用した。  

リサイズが終わったら、学習用とテスト用のディレクトリに画像を保存する。
![](/img/createDataset002.png)  

### 画像の読み込み  
学習のために、用意したデータセットの画像を読み込む  
読み込むためのSampleコード  
```python
# coding: utf-8

from PIL import Image
import numpy as np
import os
# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle

def _load_label(file_dir):
    data_files = os.listdir(file_dir)
    labels = []
    for file in data_files:
        l = file.split('-')[0]
        if(l == 'stop'):
            labels.append(0)
        elif(l == 'limitspeed'):
            labels.append(1)

    print("Load label : Done!")

    return np.array(labels)

def _load_img(file_dir,convert_type='L'):
    data_files = os.listdir(file_dir)
    imgs = []
    for file in data_files:
        img = np.frombuffer(np.array(Image.open(file_dir+'/'+file).convert(convert_type)),dtype=np.uint8)
        imgs.append(img)
    print("Load img : Done!")

    return np.array(imgs)

def _change_one_hot_label(X):
    #2のところは分類したいクラスの数を指定すること
    T = np.zeros((X.size, 2))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_dataset(DIR_PATH,convert_type='L',normalize=True,flatten=True,one_hot_label=False):
    labels = _load_label(DIR_PATH)
    imgs = _load_img(DIR_PATH,convert_type)

    if normalize:
        imgs = imgs.astype(np.float32)
        imgs /= 255.0

    if not flatten:
        #64のところは用意した画像サイズの値を指定すること
        if(convert_type == 'L'):
            imgs = imgs.reshape(-1,1,64,64)
        elif(convert_type == 'RGB'):
            imgs = imgs.reshape(-1,3,64,64)


    if one_hot_label:
        labels = _change_one_hot_label(labels)        

    return imgs,labels
```

使用例  
```python
x_train,t_train = load_dataset('./train_dataset',convert_type='RGB',flatten=True,normalize=True,one_hot_label=True)
x_test,t_test = load_dataset('./test_dataset',convert_type='RGB',flatten=True,normalize=True,one_hot_label=True)
```
返り値は2つ返ってくるので2つ用意すること  

|引数|意味|
|:--|:--|
|'./train_dataset'|学習、テスト用のディレクトリを指定|
|convert_type='RGB'|学習をグレースケールで、もしくはカラーで行うかの指定。defaultではグレースケールのconvert_type='L'|
|flatten=True|平坦化を行うかの指定。defaultではTrue|
|normalize=True|正規化を行うかの指定。(データの値が0.0~1.0の範囲に収まるようにする) defaultではTrue|
|one_hot_label=True|Onehot表現にするかの指定。defaultではTrue|
