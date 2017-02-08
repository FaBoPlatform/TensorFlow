# RaspberryPiにTensorFlowをインストールする

## 手順

1. TensorFlowに必要なモジュールのインストール
2. wheelをダウンロードしてpipでインストール
3. mockの再インストール

## 実行環境

* Raspberry Pi3 Model B
* Raspbian Jessie Lite 2016-09-23
* Python 2.7.9

## 実行

Tensorflow 0.12.1をインストールする

他バージョンのTensorFlowが必要な場合は、以下のページを参考にしてダウンロードするwheelファイルを変更する

* https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases

```
$ sudo apt-get update
$ sudo apt-get install python-pip python-dev
$ wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v0.12.1/tensorflow-0.12.1-cp27-none-linux_armv7l.whl
$ sudo pip install tensorflow-0.12.1-cp27-none-linux_armv7l.whl
$ sudo pip uninstall mock
$ sudo pip install mock
```

## TensorFlowの実行

試しにモジュールのバージョンを表示させる

```
$ python
Python 2.7.9 (default, Sep 17 2016, 20:26:04)
[GCC 4.9.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.VERSION
'0.12.1'
```

# 参考

* [samjabrahams/tensorflow-on-raspberry-pi](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
