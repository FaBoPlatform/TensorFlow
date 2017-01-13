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

```shell
$ easy_install pip
$ easy_install --upgrade six
$ pip install --ignore-installed tensorflow
```

