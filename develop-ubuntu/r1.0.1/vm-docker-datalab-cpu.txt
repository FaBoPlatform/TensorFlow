UbuntuOSインストール

# OSインストール後、アップデートする
apt-get upgrade
apt-get update
apt-get dist-upgrade

########################################
# dockerをインストールする
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


########################################
# Docker image datalab版
########################################
# 予めGoogle Cloud Platformでプロジェクト作成を終わらせておくこと
# datalab版
# https://cloud.google.com/ml/docs/how-tos/getting-set-up?hl=ja
# datalab版: gcr.io/cloud-datalab/datalab:local
# git: https://github.com/googledatalab/datalab
# git版からdocker imageを作る: https://github.com/googledatalab/datalab/wiki/Getting-Started
docker pull gcr.io/cloud-datalab/datalab:local
docker run -it -p 8081:8080 gcr.io/cloud-datalab/datalab:local
cd
curl https://raw.githubusercontent.com/GoogleCloudPlatform/cloudml-samples/master/tools/setup_docker.sh | bash
gcloud auth login
# URLが表示されるのでリンクに飛んでコードを取得し、ターミナルに張り付ける


gcloud config set project myproject-123456

root@04065bfe6d0a:~# curl https://raw.githubusercontent.com/GoogleCloudPlatform/cloudml-samples/master/tools/check_environment.py | python
####################
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  3557  100  3557    0     0  84912      0 --:--:-- --:--:-- --:--:-- 86756
WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
Your active configuration is: [default]

Success! Your environment is configured correctly.
####################

gcloud ml-engine init-project

# Google Cloud Storageにバケットを作成する(一度作ったことがあれば不要)
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-ml
gsutil mb -l ASIA-EAST1 gs://$BUCKET_NAME

# ここまででdatalabのアカウント設定終了
# project_idが必要だった。DeepFishingプロジェクトを使った。

# pipを更新しておく
pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U


# 一度dockerを抜けてコンテナを停止し、cpu/tensorflowとしてimageを作っておく
docker ps -a
docker stop 04065bfe6d0a
docker commit 04065bfe6d0a lab/tensorflow

# localhostの/home/ubuntu/notebooksをdockerの/notebooksにマウントするため、ディレクトリを作る
mkdir -p /home/ubuntu/notebooks

# datalab版dockerコンテナ起動
docker run -itd -v /home/ubuntu/notebooks:/content/datalab/notebooks -e DATALAB_DEBUG=true -p 8081:8080 lab/tensorflow 

# jupyterは http://localhost:8081
# datalabのjupyterはdockerコンテナでは8080で動作しているが、sign-inのauthコールバックがlocalhost:8081になっているため、localhost:8081をdocker:8080に転送して使う
# tensorboardも入っているけれども、実行はCloud MLにするのでここでは起動していない

