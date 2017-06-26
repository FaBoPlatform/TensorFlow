# AWS Ubuntu 16.04 - TensorFlor GPU/XLA Python/C++
PythonとC++の両方を使えるようにします
インスタンス準備は
[AWS EC2 p2.xlarge Docker (GPU)](../r1.0.1/aws-ec2-docker-gpu.md)
と同じで、Dockerイメージから異なります。
## AWS EC2 p2.xlarge Docker (GPU)と同じ部分
NVIDIA Dockerが使える状態にあるなら不要です。
```
AWS Ubuntu 16.04
kernelは
root@ip-172-21-2-8:/home/ubuntu# uname -a
Linux ip-172-21-2-8 4.4.0-72-generic #93-Ubuntu SMP Fri Mar 31 14:07:41 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux
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
## ここから新しい部分
```
#######################################
# TensorFlow Docker インストール
########################################
# https://hub.docker.com/r/tensorflow/tensorflow/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/README.md
# 公式CPU版コード付き docker: tensorflow/tensorflow:latest-develop
# 公式GPU版コード付き docker: tensorflow/tensorflow:latest-develop-gpu
# dockerコマンドはaliasを.bashrcに書いておくとよい
# alias docker='nvidia-docker'
docker pull tensorflow/tensorflow:latest-devel-gpu
nvidia-docker run -it -p 6006:6006 -p 8888:8888 tensorflow/tensorflow:latest-devel-gpu

# dockerにbashでログインした状態
apt-get update
apt-get install vim
# pipフル更新
pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

# docker内でjupyter home dirを変更する
# そのままだと/がホームディレクトリになっていてやばいので変更する
# さらにBlocking Cross Origin WebSocket Attempt.が出てkernelに接続できない状態が発生することを防ぐために、Origin通過許可を追加する
vi /root/.jupyter/jupyter_notebook_config.py
c.NotebookApp.ip = '*'
c.NotebookApp.notebook_dir = u'/notebooks/'
c.NotebookApp.allow_origin="http://MY-DOMAIN.com:8888"

# exitでdockerを抜け、docker commitでイメージを作成する
nvidia-docker ps -a
nvidia-docker commit コンテナID gpu/tensorflow
nvidia-docker rm コンテナID # ゴミ削除
mkdir -p /home/ubuntu/notebooks/logs # すでに作ってあれば不要
# 作成したdockerイメージからコンテナを作成し起動する
# jupyterが5.0になってデフォルト値が変わったのでオプションを付けて起動する
nvidia-docker run -itd -v /home/ubuntu/notebooks:/notebooks -e "PASSWORD=MYPASSWORD" -p 6006:6006 -p 8888:8888 gpu/tensorflow /bin/bash -c "tensorboard --logdir=/notebooks/logs& /run_jupyter.sh --allow-root --NotebookApp.iopub_data_rate_limit=10000000"
```
ここまでで通常のTensorFlow GPU版が動作完了。
## gitソースコードをbuildする
```
# dockerにbashでログインする
nvidia-docker ps -a
nvidia-docker exec -it コンテナID /bin/bash
cd /tensorflow
git pull
# XLA just-in-time compiler はお好みで、それ以外はデフォルトで
./configure
Please specify the location of python. [Default is /usr/bin/python]: 
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
Do you wish to use jemalloc as the malloc implementation? [Y/n] 
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] 
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] 
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] Y
XLA JIT support will be enabled for TensorFlow
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]

Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with OpenCL support? [y/N] 
No OpenCL support will be enabled for TensorFlow
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 
Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify the Cudnn version you want to use. [Leave empty to use system default]: 
Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Extracting Bazel installation...
..........


####################
# GPU版ビルド
####################
# AWS EC2 p2.xlargeインスタンス
time bazel build -c opt --config=cuda --verbose_failures //tensorflow/tools/pip_package:build_pip_package

real	95m46.429s
user	0m0.220s
sys	0m0.388s

# whlを生成
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# TensorFlow インストール
pip install /tmp/tensorflow_pkg/tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl
# .soを作成
time bazel build -c opt --config=cuda --verbose_failures //tensorflow:libtensorflow_cc.so

real	4m25.759s
user	0m0.016s
sys	0m0.024s

# /usr/local/lib にインストールする
install -m 0644 bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/
ldconfig
```
## ヘッダーファイルをインストールする
このスクリプトを/tensorflow/直下で実行する

参考：http://memo.saitodev.com/home/tensorflow/build/
```
#!/bin/bash -eu
# -*- coding: utf-8 -*-
# tensorflowを利用するC++ソースをコンパイルする際に、このディレクトリをインクルードパスに追加する。

HEADER_DIR=/usr/local/tensorflow/include

if [ ! -e $HEADER_DIR ];
then
    mkdir -p $HEADER_DIR
fi

find tensorflow/core -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
find tensorflow/cc   -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
find tensorflow/c    -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;

find third_party/eigen3 -follow -type f -exec cp --parents {} $HEADER_DIR \;

pushd bazel-genfiles
find tensorflow -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
popd

pushd bazel-tensorflow/external/protobuf/src
find google -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
popd

pushd bazel-tensorflow/external/eigen_archive
find Eigen       -follow -type f -exec cp --parents {} $HEADER_DIR \;
find unsupported -follow -type f -exec cp --parents {} $HEADER_DIR \;
popd
```
## チュートリアルコンパイル
```
# チュートリアルのコンパイル
time bazel build -c opt //tensorflow/cc:tutorials_example_trainer

# チュートリアルの実行
time ./bazel-bin/tensorflow/cc/tutorials_example_trainer
```
## 自前C++コードのコンパイル
```
########################################
# コンパイル方法
########################################
# loadGraph.cpp
g++ loadGraph.cpp -std=c++11 -I/usr/local/tensorflow/include -L. -ltensorflow_cc -Wl,-rpath=.
```
## Dockerイメージ作成
作成したコンテナをイメージに保存しておく
```
nvidia-docker ps -a
nvidia-docker commit コンテナID gpu/tensorflow
nvidia-docker rm コンテナID
```
## Dockerコンテナ作成-起動
```
nvidia-docker run -itd -v /home/ubuntu/notebooks:/notebooks -e "PASSWORD=MYPASSWORD" -p 6006:6006 -p 8888:8888 gpu/tensorflow /bin/bash -c "tensorboard --logdir=/notebooks/logs& /run_jupyter.sh --allow-root --NotebookApp.iopub_data_rate_limit=10000000"
```
## Dockerコンテナ停止-起動
通常の起動と停止
```
nvidia-docker stop コンテナID
nvidia-docker start コンテナID
```