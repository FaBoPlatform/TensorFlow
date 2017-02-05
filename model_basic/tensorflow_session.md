# セッションの保存・読み込み

セッション内の`Variables`を保存・読み込む。モデルの学習時にパラメタの値を保存しておきたい場合に利用する。

## Sample

![](/img/session01.png)

![](/img/session02.png)

以下のようなファイルが生成させる

```shell
$ ls rand*
rand-0.data-00000-of-00001 rand-1.data-00000-of-00001 rand-2.data-00000-of-00001
rand-0.index               rand-1.index               rand-2.index
rand-0.meta                rand-1.meta                rand-2.meta
```

## ノート

セッションの読み込み時にファイルパスを"ファイル名"とすると失敗する可能性あり。"./ファイル名"としたところ成功した。

実行環境:

* Python 2.7.12 :: Anaconda 4.1.1 (x86_64)
* TensorFlow 0.12.0-rc0

類似の不具合:

* https://github.com/tensorflow/tensorflow/issues/571

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/session.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/session.ipynb)


