# Jetson TX2 TensorFlow r1.3.0 ビルド方法
## JetPack 3.1 Python3.6.2/TensorFlow r1.3.0/OpenCV3.2/OpenMPI2.1.1
##### Ubuntu 16.04 LTS - TensorFlow r1.3.0

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
  * OpenCV-3.2コンパイル
  * bazel-0.5.3コンパイル
  * OpenMPI-2.1.1コンパイル
  * TensorFlow-r1.3.0コンパイル




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
# OpenCV 3.2 ビルド
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

mkdir -p /compile \
&& cd /compile \
&& wget --no-check-certificate https://github.com/opencv/opencv/archive/3.2.0.zip \
&& unzip 3.2.0.zip \
&& mkdir -p opencv-3.2.0/build \
&& cd opencv-3.2.0/build

cmake -DCMAKE_C_FLAGS="-march=native" -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D FORCE_VTK=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D CUDA_CUDA_LIBRARY=/usr/lib/aarch64-linux-gnu/libcuda.so -D WITH_GDAL=ON -D WITH_XINE=ON -D BUILD_EXAMPLES=ON -D PYTHON3_EXECUTABLE=/usr/bin/python3 -D PYTHON_INCLUDE_DIR=/usr/include/python3.6 -D PYTHON_INCLUDE_DIR2=/usr/include/aarch64-linux-gnu/python3.6m -D PYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.6/dist-packages/numpy/core/include/ ..

make install -j $(($(nproc) + 1))

mkdir -p /etc/ld.so.conf.d
echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf
ldconfig
```

```
########################################
# bazel-0.5.3 ビルド
########################################
mkdir -p /compile \
 cd /compile \
&& wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.5.3/bazel-0.5.3-dist.zip \
&& unzip bazel-0.5.3-dist.zip -d bazel-0.5.3 \
&& cd bazel-0.5.3 \
&& ./compile.sh \
&& cp output/bazel /usr/local/bin/
```

```
#######################################
# OpenMPI ビルド
########################################
apt-get remove -y openmpi-common libopenmpi1.10

mkdir -p /compile \
&& cd /compile \
&& wget --no-check-certificate https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.gz \
&& tar -zxvf openmpi-2.1.1.tar.gz \
&& cd openmpi-2.1.1 \
&& ./configure --prefix=/usr --enable-mpi-thread-multiple --with-cuda=/usr/local/cuda-8.0/targets/aarch64-linux --with-cuda-libdir=/usr/lib/aarch64-linux-gnu CXXFLAGS='-std=c++11 -march=native' \
&& make install
```

```
########################################
# TensorFlow r1.3.0
########################################
# TensorFlow r1.3.0 download
mkdir -p /compile \
&& cd /compile \
&& git clone -b r1.3 https://github.com/tensorflow/tensorflow


#eigenはARM NEONにバグがあるので修正版を入れる
vi /compile/tensorflow/tensorflow/workspace.bzl
  native.new_http_archive(
      name = "eigen_archive",
      urls = [
#          "http://mirror.bazel.build/bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
#          "https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
          "https://bitbucket.org/eigen/eigen/get/d781c1de9834.tar.gz",
      ],
#      sha256 = "ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4",
#      strip_prefix = "eigen-eigen-f3a22f35b044",
      strip_prefix = "eigen-eigen-d781c1de9834",
      build_file = str(Label("//third_party:eigen.BUILD")),
  )

# github内容が書き換えられてprotobuf/llvmのchecksumが通らなくなった対策としてsha256をコメントアウト
# https://github.com/tensorflow/tensorflow/issues/12979
vi /compile/tensorflow/tensorflow/workspace.bzl

#      sha256 = "00fb4a83a4dd1c046b19730a80e2183acc647715b7a8dcc8e808d49ea5530ca8",
#      sha256 = "6d43b9d223ce09e5d4ce8b0060cb8a7513577a35a64c7e3dad10f0703bf3ad93",


# build/install Python API
cd /compile/tensorflow \
&& { echo;echo;echo "Y";echo;echo;echo;echo;echo;echo "Y";echo;echo;echo "Y";echo;echo;echo;echo;echo;echo;echo 6.2;echo "Y";echo;cat /dev/stdin; } | ./configure

bazel build --config=cuda -c opt --copt='-march=native' --verbose_failures --subcommands //tensorflow/tools/pip_package:build_pip_package \
&& bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
&& pip3 install /tmp/tensorflow_pkg/tensorflow-1.3.0-cp36-cp36m-linux_aarch64.whl
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