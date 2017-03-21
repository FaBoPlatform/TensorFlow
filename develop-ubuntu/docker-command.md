# Docker Commandについて

対象OS:Ubuntu。すべてrootで実行。
##### dockerをインストールしておく必要があります。
```
########################################
# Docker install
# https://docs.docker.com/engine/installation/linux/ubuntu/
########################################
# curl他をインストールする
apt-get install curl linux-image-extra-$(uname -r) linux-image-extra-virtual
# docker repositoryをインストールする
apt-get install apt-transport-https ca-certificates
# docker GPG keyをインストールする
curl -fsSL https://yum.dockerproject.org/gpg | sudo apt-key add -
# dockerキーを確認する
apt-key fingerprint 58118E89F3A912897C070ADBF76221572C52609D

# docker stable repository
add-apt-repository "deb https://apt.dockerproject.org/repo/ ubuntu-$(lsb_release -cs) main"

apt-get update
# dockerをインストールする
apt-get install docker-engine
# dockerバージョンリスト
apt-cache madison docker-engine
# docker動作確認
docker run hello-world
```

##### 利用できるdockerイメージを検索
```
########################################
# dockerイメージを検索
########################################
docker search tensorflow
```

##### Dockerイメージをダウンロード
```
########################################
# TensorFlow Docker image download
########################################
# https://hub.docker.com/r/tensorflow/tensorflow/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/README.md
# 公式版docker: tensorflow/tensorflow
# google cloudが出しているdocker: gcr.io/tensorflow/tensorflow
docker pull tensorflow/tensorflow
```

##### Dockerコンテナをイメージから作成し起動する
```
########################################
# TensorFlow Docker launch from image (make new docker container=process)
########################################
# localhostの/home/ubuntu/notebooksをdockerの/notebooksにマウントするため、ディレクトリを作る
# tensorboard用のディレクトリを適当に作る
mkdir -p /home/ubuntu/notebooks/logs

# Tensorflow公式Dockerのport番号
# jupyter port:8888, tensorboard port:6006
docker run -itd -v /home/ubuntu/notebooks:/notebooks "PASSWORD=mypassword" -p 6006:6006 -p 8888:8888 tensorflow/tensorflow /bin/bash -c "tensorboard --logdir=/notebooks/logs& /run_jupyter.sh"
docker ps -a
# jupyterへのアクセス方法:ブラウザで http://localhost:8888
# passwordは起動時に設定したmypassword
# pullしていない場合はダウンロードから起動まで実行になる
# tensorflow/tensorflowはTensorFlow公式CPU版Docker
# 自分でpip install等をした場合は別レポジトリにcommitしてイメージを作成して使う
```

##### Dockerコンテナをcontainer_idから起動する
container_id:437620a48fb7
```
########################################
# Dockerコンテナを起動
########################################
docker ps -a
docker start 437620a48fb7
```

##### 起動中のDockerコンテナにbashでログインする
container_id:437620a48fb7
```
########################################
# Dockerコンテナにbashでログイン
########################################
docker ps -a
docker exec -it 437620a48fb7 /bin/bash
```

##### Dockerコンテナを停止する
container_id:437620a48fb7
```
########################################
# Dockerコンテナを停止
########################################
docker ps -a
docker stop 437620a48fb7
```

##### Dockerコンテナを削除する
container_id:437620a48fb7
```
########################################
# Dockerコンテナを削除
########################################
docker ps -a
docker rm 437620a48fb7
```

##### Dockerイメージを削除する
image_id:cd88548da3c5
```
########################################
# Dockerイメージを削除
########################################
docker images
docker rmi cd88548da3c5
```

##### Dockerイメージを作成する
Dockerコンテナを停止しておくこと
container_id:437620a48fb7
```
########################################
# Dockerイメージを作成
########################################
docker ps -a
docker commit 437620a48fb7 cpu/tensorflow
# コンテナにbashでログインしてpip install等を行ったコンテナは、OSを再起動しても消えないけれどもイメージとして作成しておくとよい
```

##### Dockerイメージを最新に更新する
pip install等はやり直しになるが、自分で書いたコードはdocker run -itd -v /home/ubuntu/notebooks:/notebooksで指定した/home/utuntu/notebooks/以下に保存してある
```
########################################
# Dockerイメージ最新に更新
########################################
docker images | cut -d ' ' -f1 | tail -n +2 | sort | uniq | egrep -v '^(<none>|ubuntu)$' | xargs -P0 -L1 sudo docker pull
```