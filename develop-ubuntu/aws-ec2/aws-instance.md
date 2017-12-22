# AWS INSTANCE OS INFO
\# AWS Ubuntu 16.04 64bit
\# 普通のUbuntuを使う
\# Ubuntu Server 16.04 LTS (HVM), SSD Volume Type - ami-aa2ea6d0
\# （今時のAWSはDeep Learning AMI(Ubuntu/Amazon Linux)が提供されている）

- OSインストール後、アップデートする
\# Package configurationはinstall the package maintainer's versionを選択した
> - apt-get update
> - apt-get upgrade -y
> - apt-get dist-upgrade -y

\# https://docs.docker.com/engine/installation/linux/ubuntu/
- curl他をインストールする
> - apt-get install -y curl linux-image-extra-$(uname -r) linux-image-extra-virtual

- 認証パッケージをインストールする
> - apt-get install -y apt-transport-https ca-certificates

- docker GPG keyをインストールする
> - curl -fsSL https://yum.dockerproject.org/gpg | sudo apt-key add -

- dockerキーを確認する
> - apt-key fingerprint 58118E89F3A912897C070ADBF76221572C52609D

- docker stable repositoryを登録する
> - add-apt-repository "deb https://apt.dockerproject.org/repo/ ubuntu-$(lsb_release -cs) main"
> - apt-get update

- dockerをインストールする
> - apt-get install -y docker-engine

- dockerバージョンリストを確認する
> - apt-cache madison docker-engine

- docker動作確認
> - docker run --rm hello-world

- Setup Nvidia Drivers on EC2 Instance Host
\# https://github.com/fluxcapacitor/pipeline/wiki/AWS-GPU-TensorFlow-Docker
> - apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
> - sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
> - apt-get update && apt-get install -y --no-install-recommends cuda-drivers

- NVIDIA dockerをインストールする for Ubuntu 16.04 LTS
\# https://github.com/NVIDIA/nvidia-docker/
\# Install nvidia-docker and nvidia-docker-plugin
> - wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
> - dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

- Test nvidia-smi
> - nvidia-docker run --rm nvidia/cuda nvidia-smi

- 不要になったパッケージを削除
> - apt autoremove -y

- reboot
> - reboot


\# uninstall nvidia docker
\#apt-get purge -y nvidia*
\#apt-get purge -y cuda*