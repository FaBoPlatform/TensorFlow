# Google Cloud DataLabのTensorFlowをUpdate
（この項目はTensorFlow r1.0リリース前に、最新版のTensorFlowに更新した際の情報です。ライブラリの更新が必要な時に参考にしてください)

チュートリアルに必要なTensorFlowのVersionは、1.0.0です。

## Dockerコンテナ一覧

`docker ps -a`でDockerコンテナの一覧を取得する。(詳細はman docker-ps)

```shell
$ docker ps -a
```
Dockerコンテナとは：Dockerイメージをdocker runすることで作られるDockerインスタンスのこと。dockerが実行するプロセスイメージ。docker stop/startで停止/起動が可能。

## Consoleへログイン

取得したNameもしくは、コンテナIDを指定して、ログインする。


[コンテナIDで起動IDで起動]
```shell
$ docker exec -it コンテナID /bin/bash
```

[Nameで起動]
```shell
$ docker exec -it MAME /bin/bash
```

## PythonのVersion

ConsoleにログインしたらPythonのVersionを調べる。

```shell
$ python -V
Python 2.7.9
```

## TensorFlow 1.0.0へUpdate

Python 2.7系で、TensorFlow 1.0.0にUpdateする。


```shell
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0rc0-cp27-none-linux_x86_64.whl
$ pip install --ignore-installed --upgrade $TF_BINARY_URL
```
または
```shell
pip install tensorflow -U
```
## Updateの確認

一通りアップデートが終わったら
```shell
$ pip list
```
でtensorflowのバージョンを確認し、`tensorflow (1.0.0RC0)`　が見つかればUpdateされている。

## Dockerに保存

最後に、TF1.0.0RC0にUpdateされた状態でDockerを保存する。
再び、`docker ps`で、起動しているDockerコンテナ一覧を取得する。

```shell
$ docker ps
```

コンテナIDをコピーし、下記のコマンドで変更をコミットする

```shell
$ docker commit -m "tensorflow version up" 19627749df78 datalab_tf1
```

|引数|意味|
|:--|:--|
|-m "tensorflow version up" | 好きなコメントを記載|
|19627749df78| `docker ps`で確認した各自のCONTAINER ID|
|datalab_tf1|任意の名前|

コミットが無事成功したら
```shell
$ docker images
```
で好きな名前をつけたimageが作成されていることを確認する

## Datalabの再起動

DONTAINER IDを指定して、起動中のDockerを停止する。

```shell
$ docker stop 19627749df78
```

[OS Xで実行:プロジェクトIDを指定]
```shell
$ cd ~
$ mkdir -p ./datalab
$ docker run -it -p 127.0.0.1:8081:8080 -p 6006:6006 -v "${HOME}/datalab:/content" \
 -e "PROJECT_ID=プロジェクトID"  \
datalab_tf1
```

[OS Xで実行:プロジェクトIDを未指定]
```shell
$ cd ~
$ mkdir -p ./datalab
$ docker run -it -p 127.0.0.1:8081:8080 -p 6006:6006 -v "${HOME}/datalab:/content" \
datalab_tf1
```
## Consoleへログイン

取得したNameもしくは、コンテナIDを指定して、ログインする。


[コンテナIDで起動]
```shell
$ docker exec -it コンテナID /bin/bash
```

[Nameで起動]
```shell
$ docker exec -it MAME /bin/bash
```

最後に、
```shell
$ pip list
```
をし、`tensorflow (1.0.0RC0)`を確認できれば、正常にDocker Imageが生成された事になる。
