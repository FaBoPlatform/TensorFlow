# Google Cloud DataLabのTensorFlowをUpdate

チュートリアルに必要なTensorFlowのVersionは、0.12.1です。

```shell
$ docker ps
```
でNameを取得し、そのName指定で、ログインする。

```shell
$ docker exec -it datalab /bin/bash
```

```shell
$ python -V      
Python 2.7.9
```

```shell
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0a0-cp27-none-linux_x86_64.whl
$ pip install --ignore-installed --upgrade $TF_BINARY_URL
```

一通りアップデートが終わったら
```shell
$ pip list
```
でtensorflowのバージョンを確認し、`tensorflow (1.0.0a0)`　が見つかれば、
そのshell上で`exit`ではなく、Ctrl+Q Ctrl+Pでdetach（動かしたまま抜ける）する  

```shell
$ docker ps
```
で動いているのを確認しておく  

確認できたら次に下記のコマンドでコミットする
```shell
$ docker commit -m "tensorflow version up" 19627749df78 testimage
```
("tensorflow version up"の所は好きなコメント、19627749df78は`docker ps`で確認した各自のCONTAINER ID、testimageは好きな名前でいいはず)  

コミットが無事成功したら
```shell
$ docker images
```
で好きな名前をつけたimageが作成されていることを確認する

Ctrl+Q Ctrl+Pでdetachしたままのコンテナを終了して、(`$ docker stop CONTAINER_ID`で終了できる)
```shell
$ docker run -it -p "127.0.0.1:8081:8080" -v "${HOME}/datalab:/content" testimage
```
(testimageは各自の作成したimage名)で作成したものを起動する  
新しくターミナルを起動して
```shell
$ docker exec -it NAME /bin/bash
```
そのshell上で
```shell
$ pip list
```
をし、`tensorflow (0.12.1)`となっていたら終わり
