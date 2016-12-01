
# TensorFlowのインストール

OS XでのTensorFlowのインストール
https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

```shell
# Mac OS X, CPU only, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py2-none-any.whl

# Mac OS X, GPU enabled, Python 2.7:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.11.0-py2-none-any.whl
```

```shell
# Python 2
$ sudo pip install --upgrade $TF_BINARY_URL
```

0.11.0を使うと
```shell
ImportError: No module named _pywrap_tensorflow
```
のエラーが出てしまう。その場合は、

0.10.0を使う
```shell
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl
```



