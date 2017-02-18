# TensorFlowのVersion UP

本テキスト用に、TensorFlowのVersionを1.0.0にUpdateする。

```shell
$ wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl
$ pip install --user --upgrade tensorflow-1.0.0-cp27-none-linux_x86_64.whl

$ python -c 'import tensorflow as tf; print tf.__version__'
1.0.0
```