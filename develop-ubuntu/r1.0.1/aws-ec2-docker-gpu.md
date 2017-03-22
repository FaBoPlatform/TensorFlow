# AWS Ubuntu 16.04
```
AWS Ubuntu 16.04
kernelは
root@ip-172-21-2-7:/home/ubuntu# uname -a
Linux ip-172-21-2-7 4.4.0-64-generic #85-Ubuntu SMP Mon Feb 20 11:50:30 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux

# OSインストール後、アップデートする
apt-get upgrade
apt-get update
apt-get dist-upgrade
reboot
```

```
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
```

```
########################################
# Setup Nvidia Drivers on EC2 Instance Host
########################################
# https://github.com/fluxcapacitor/pipeline/wiki/AWS-GPU-TensorFlow-Docker
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

apt-get update && apt-get install -y --no-install-recommends cuda-drivers
```

```
########################################
# NVIDIA dockerをインストールする
########################################
# https://github.com/NVIDIA/nvidia-docker/
# Install nvidia-docker and nvidia-docker-plugin
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Test nvidia-smi
nvidia-docker run --rm nvidia/cuda nvidia-smi
```

```
########################################
# TensorFlow Docker インストール
########################################
# https://hub.docker.com/r/tensorflow/tensorflow/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/README.md
# 公式CPU版 docker: tensorflow/tensorflow
# 公式GPU版 docker: tensorflow/tensorflow:latest-gpu

docker search tensorflow
docker pull tensorflow/tensorflow:latest-gpu
nvidia-docker run -it -e "PASSWORD=mypassword" -p 6006:6006 -p 8888:8888 tensorflow/tensorflow:latest-gpu

# Blocking Cross Origin WebSocket Attempt.が出てkernelに接続できない状態が発生することを防ぐために、Origin通過許可を追加する
apt-get update
apt-get vim
vi /root/.jupyter/jupyter_notebook_config.py
c.NotebookApp.allow_origin="http://YOURDOMAIN:8888"

# 一度dockerを抜けてコンテナを停止し、cpu/tensorflowとしてimageを作っておく
docker ps -a
docker stop d0b399c9ff83
docker commit d0b399c9ff83 gpu/tensorflow

# localhostの/home/ubuntu/notebooksをdockerの/notebooksにマウントするため、ディレクトリを作る
# tensorboard用のディレクトリを適当に作る
mkdir -p /home/ubuntu/notebooks/logs
nvidia-docker run -itd -v /home/ubuntu/notebooks:/notebooks -e "PASSWORD=mypassword" -p 6006:6006 -p 8888:8888 gpu/tensorflow /bin/bash -c "tensorboard --logdir=/notebooks/logs& /run_jupyter.sh"

# jupyterは http://localhost:8888
# tensorboardは http://localhost:6006
```

```
########################################
# pipライブラリを入れる
########################################
pip install pandas
pip install seaborn
pip install requests
# pipフル更新
pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
```