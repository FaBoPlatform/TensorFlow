# RaspberryPiにTensorFlowをインストールする

## 手順

1. TensorFlowに必要なモジュールのインストール
2. wheelをダウンロードしてpipでインストール
3. mockの再インストール

## インストール

**Tensorflow 1.0.1**をインストールする

他バージョンのTensorFlowが必要な場合は、以下のページを参考にしてダウンロードするwheelファイルを変更する

* https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases

**Python2.7系**：

```
$ sudo apt-get update
$ sudo apt-get install python-pip python-dev
$ wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.0.1/tensorflow-1.0.1-cp27-none-linux_armv7l.whl
$ sudo pip install tensorflow-1.0.1-cp27-none-linux_armv7l.whl
$ sudo pip uninstall mock
$ sudo pip install mock
```

**Python3系**：

```
$ sudo apt-get update
$ sudo apt-get install python3-pip python3-dev
$ wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.0.1/tensorflow-1.0.1-cp34-cp34m-linux_armv7l.whl
$ sudo pip3 install tensorflow-1.0.1-cp34-cp34m-linux_armv7l.whl
$ sudo pip3 uninstall mock
$ sudo pip3 install mock
```

## TensorFlowの実行

TensorFlowのバージョンを表示させる。

**Python2.7系**：

```
$ python
Python 2.7.9 (default, Sep 17 2016, 20:26:04)
[GCC 4.9.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.VERSION
'1.0.1'
```

**Python3系**：

```
$ python3
Python 3.4.2 (default, Oct 19 2014, 13:31:11)
[GCC 4.9.1] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.VERSION
'1.0.1'
```

## アンインストール

```
# python2.7+
$ sudo pip uninstall tensorflow
# python3+
$ sudo pip uninstall tensorflow
```

## 実行環境

* Raspberry Pi3 Model B
* Raspbian Jessie Lite 2016-09-23
* Python 2.7.9
* Python 3.4.2

# 参考

* [samjabrahams/tensorflow-on-raspberry-pi](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
