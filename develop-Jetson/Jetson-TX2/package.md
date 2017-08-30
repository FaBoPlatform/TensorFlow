# Jetson TX2 TensorFlow r1.3.0 packageインストール
## JetPack 3.1 Python3.6.2/TensorFlow r1.3.0/OpenCV3.2/OpenMPI2.1.1
##### Ubuntu 16.04 LTS - TensorFlow r1.3.0
OpenCV3.2、OpenMPI-2.1.1、TensorFlow-1.3.0をパッケージ化したものを導入する。

パッケージは[binary](./binary)以下にあるものを使う。
Python環境、特にnumpyバージョンがシビアなので、パッケージ版でも環境準備にはビルド時と同じ手順を踏む必要がある。

# 流れ
* [JetPack 3.1インストール](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.1/jetpack_l4t_install.htm)
* CPUファン起動
* コンパイル
  * パッケージ更新
  * bash環境変数追加
  * レポジトリ登録
  * Python-3.6インストール
  * Python-3.6設定
  * pipパッケージインストール
  * Jupyter設定
  * Java8インストール
  * パッケージインストール
  * locateファイルパスデータベース登録
  * CUDA demoQueryコンパイル
  * パッケージインストール準備
  * OpenCV-3.2 パッケージインストール
  * OpenMPI-2.1.1 パッケージインストール
  * TensorFlow-r1.3.0 パッケージインストール




Ubuntu確認
```
cat /etc/lsb-release

DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=16.04
DISTRIB_CODENAME=xenial
DISTRIB_DESCRIPTION="Ubuntu 16.04 LTS"
```
カーネル確認
```
uname -a

Linux tegra-ubuntu 4.4.38-tegra #1 SMP PREEMPT Thu Jul 20 00:49:07 PDT 2017 aarch64 aarch64 aarch64 GNU/Linux
```
アーキテクチャ確認
```
uname -m

aarch64
```

pip install用ビルド済みバイナリ
[TensorFlow r1.3.0(GPU)](./binary/tensorflow-1.3.0-cp36-cp36m-linux_aarch64.whl)

# TensorFlow build on Jetson TX2 JetPack 3.1
```
########################################
# JetPack L4T 3.1 (Ubuntu 16.04.02に更新)
########################################
http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.1/jetpack_l4t_install.htm

公式インストールマニュアルに従いインストールを行う。
NVIDIAにユーザ登録し、JetPack-L4T-3.1-linux-x64.runをダウンロードする必要がある。

以下、JetPack L4T 3.1 更新済み。
```

```
########################################
# CPUファン起動
########################################
sh -c 'echo 255 > /sys/kernel/debug/tegra_fan/target_pwm'
```

```
########################################
# CPUファン起動設定
########################################
cat <<EOF> /etc/init.d/cpufan
#!/bin/sh
### BEGIN INIT INFO
# Provides:         cpufan
# Required-Start:   $remote_fs $syslog
# Required-Stop:    $remote_fs $syslog
# Default-Start:    2 3 4 5
# Default-Stop:	    0 1 6
# Short-Description: CPU Fan launcher
### END INIT INFO

# Launch CPU Fan
sh -c 'echo 255 > /sys/kernel/debug/tegra_fan/target_pwm'
EOF

chmod 755 /etc/init.d/cpufan
update-rc.d cpufan defaults
```

```
########################################
# Ubuntu 16.04 パッケージ更新
########################################
apt-get update
apt-get upgrade
apt-get dist-upgrade
apt-get install htop
apt autoremove
```


```
########################################
# .bashrc (ubuntu/root両方)
########################################
-    alias ls='ls --color=auto'
+    alias ls='ls -asiF --color=auto'

export PATH=/usr/local/cuda-8.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/lib:
export __GL_PERFMON_MODE=1
export LANG="en_US.UTF-8"
export LC_ALL=$LANG
export LC_CTYPE=$LANG
```

```
########################################
# Python3.6, Java8 レポジトリ追加
########################################
apt-get install -y software-properties-common
{ echo;cat /dev/stdin; } | add-apt-repository ppa:jonathonf/python-3.6
add-apt-repository "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main"
{ echo;cat /dev/stdin; } | add-apt-repository ppa:webupd8team/y-ppa-manager
apt-get update
```

```
########################################
# Python環境構築
########################################
apt-get install -y python3.6 python3.6-dev
update-alternatives --install /usr/bin/python3 python /usr/bin/python3.6 0
apt-get install -y python3-pip
```

```
########################################
# ENV Python3.6設定
########################################
rm -rf /usr/bin/python
ln -s /usr/bin/python3.6 /usr/bin/python
```

```
########################################
# pip install
########################################
apt-get install libjpeg-dev libxslt-dev libxml2-dev libffi-dev libcurl4-openssl-dev libssl-dev libblas-dev liblapack-dev gfortran

pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --upgrade numpy
pip3 install --upgrade scipy
pip3 install --upgrade pandas
pip3 install --upgrade matplotlib
pip3 install --upgrade seaborn
pip3 install --upgrade requests
pip3 install --upgrade futures
pip3 install --upgrade Pillow
pip3 install --upgrade jupyter
pip3 install --upgrade sklearn
pip3 install --upgrade tqdm
pip3 install --upgrade scikit-image

pip3 freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
```

```
########################################
# Jupyter setting
########################################
mkdir -p /home/ubuntu/notebooks
jupyter notebook --generate-config --allow-root

echo -e "import os\n\
from IPython.lib import passwd\n\
\n\
c.NotebookApp.ip = '*'\n\
c.NotebookApp.notebook_dir = u'/home/ubuntu/notebooks/'\n\
c.NotebookApp.port = int(os.getenv('PORT', 8888))\n\
#c.NotebookApp.allow_origin='http://localhost:8888'\n\
c.NotebookApp.open_browser = False\n\
c.MultiKernelManager.default_kernel_name = 'python3'\n\
\n\
# sets a password if PASSWORD is set in the environment\n\
if 'PASSWORD' in os.environ:\n\
  c.NotebookApp.password = passwd(os.environ['PASSWORD'])\n\
  del os.environ['PASSWORD']\n"\
>> /root/.jupyter/jupyter_notebook_config.py
```

```
########################################
# Java8 インストール
########################################
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections && apt-get install -y oracle-java8-installer
```

```
########################################
# Build Tools Install
########################################
apt-get install -y zip git curl locate libeigen3-dev libprotobuf-dev

updatedb
# updatedb時にパーミッションエラーが発生するため、エラーが発生したパスをumountする
umount /run/user/106/gvfs
updatedb
locate libcuda.so
```

```
########################################
# CUDA demoQueryビルド
########################################
cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
make
mkdir -p /usr/local/cuda/extras/demo_suite
ln -s /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery/deviceQuery /usr/local/cuda/extras/demo_suite
```

```
########################################
# パッケージ作成/インストール準備
########################################
dpkg --print-foreign-architectures
dpkg --add-architecture aarch64
dpkg --print-foreign-architectures
```

```
########################################
# OpenCV 3.2 パッケージインストール
########################################
apt-get install -y build-essential cmake libeigen3-dev libatlas-base-dev gfortran git wget libavformat-dev libavcodec-dev libswscale-dev libavresample-dev ffmpeg pkg-config unzip qtbase5-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev v4l-utils liblapacke-dev libopenblas-dev checkinstall libgdal-dev

# https://devtalk.nvidia.com/default/topic/1007290/building-opencv-with-opengl-support-/
# #error Please include the appropriate gl headers before including cuda_gl_interop.h
# make中にエラーが発生するため、cudaヘッダを書き換える
vi /usr/local/cuda-8.0/include/cuda_gl_interop.h
####################
#else /* __APPLE__ */

//#if defined(__arm__) || defined(__aarch64__)
//#ifndef GL_VERSION
//#error Please include the appropriate gl headers before including cuda_gl_interop.h
//#endif
//#else
#include <GL/gl.h>
//#endif

#endif /* __APPLE__ */
####################
cd /usr/lib/aarch64-linux-gnu/
ln -sf tegra/libGL.so libGL.so

# パッケージダウンロード
wget --no-check-certificate https://github.com/FaBoPlatform/TensorFlow/tree/master/develop-Jetson/Jetson-TX2/binary/opencv-3.2.deb
dpkg -i opencv-3.2.deb

ldconfig
```

```
#######################################
# OpenMPI パッケージインストール
########################################
apt-get remove -y openmpi-common libopenmpi1.10

# パッケージダウンロード
wget --no-check-certificate https://github.com/FaBoPlatform/TensorFlow/tree/master/develop-Jetson/Jetson-TX2/binary/openmpi-2.1.1.deb
dpkg -i openmpi-2.1.1.deb
```

```
########################################
# TensorFlow r1.3.0
########################################
# TensorFlow r1.3.0 Python API
wget --no-check-certificate https://github.com/FaBoPlatform/TensorFlow/tree/master/develop-Jetson/Jetson-TX2/binary/tensorflow-1.3.0-cp36-cp36m-linux_aarch64.whl
pip3 install tensorflow-1.3.0-cp36-cp36m-linux_aarch64.whl

# TensorFlow r1.3.0 C++ API
wget --no-check-certificate https://github.com/FaBoPlatform/TensorFlow/tree/master/develop-Jetson/Jetson-TX2/binary/tensorflow-cpp-1.3.0.deb
dpkg -i tensorflow-cpp-1.3.0.deb
```

# Jupyter 起動方法
```
env PASSWORD=mypassword jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000
```
# Jupyter アクセス方法
```
ブラウザで
http://IPアドレス:8888/
パスワードは起動時に環境変数に指定したmypassword
```