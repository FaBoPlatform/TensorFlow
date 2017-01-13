# Google Cloud DataLabのTensorFlowをUpdate

チュートリアルに必要なTensorFlowのVersionは、0.12.1です。

```shell
$ docker ps
```
でNameを取得し、そのName指定で、ログインする。

```shell
$ ocker exec -it datalab /bin/bash
```

```shell
$ python -V      
Python 2.7.9
```

https://www.tensorflow.org/get_started/os_setup を参考にpipコマンドでインストールする。

```shell
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py2-none-any.whl
$ sudo pip install --upgrade $TF_BINARY_URL
```

