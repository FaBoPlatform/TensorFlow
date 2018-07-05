########################################
# OpenCV 3.3.1 パッケージ作成/インストール
########################################
SCRIPT_DIR=$(cd $(dirname $0); pwd)
# cudaコンパイラのnvccは-std=c++11を扱えないのでFLAGを立てないこと

apt-get install -y build-essential cmake libeigen3-dev libatlas-base-dev gfortran git wget libavformat-dev libavcodec-dev libswscale-dev libavresample-dev ffmpeg pkg-config unzip qtbase5-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev v4l-utils liblapacke-dev libopenblas-dev checkinstall libgdal-dev


mkdir -p /compile \
&& cd /compile \
&& wget --no-check-certificate https://github.com/opencv/opencv/archive/3.3.1.zip \
&& unzip 3.3.1.zip \
&& mkdir -p opencv-3.3.1/build \
&& cd opencv-3.3.1/build

time cmake -D CMAKE_C_FLAGS="-std=c11 -march=native" -D CMAKE_CXX_FLAGS="-march=native" -D ENABLE_CXX11=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/package_build/opencv-3.3.1/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.2" -D CUDA_ARCH_PTX="6.2" -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D CUDA_CUDA_LIBRARY=/usr/lib/aarch64-linux-gnu/libcuda.so -D WITH_GDAL=ON -D WITH_XINE=ON -D BUILD_EXAMPLES=ON -D PYTHON3_EXECUTABLE=/usr/bin/python3 -D PYTHON_INCLUDE_DIR=/usr/include/python3.6 -D PYTHON_INCLUDE_DIR2=/usr/include/aarch64-linux-gnu/python3.6m -D PYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.6/dist-packages/numpy/core/include/ ..

time make install -j $(($(nproc) + 1))

mkdir -p /package_build/opencv-3.3.1/etc/ld.so.conf.d
echo "/package_build/opencv-3.3.1/usr/local/lib" > /package_build/opencv-3.3.1/etc/ld.so.conf.d/opencv.conf


mkdir -p /package_build/opencv-3.3.1/DEBIAN \
&& cd /package_build \
&& echo -e "Source: opencv-3.3.1\n\
Package: opencv\n\
Version: 3.3.1\n\
Priority: optional\n\
Maintainer: Yoshiroh Takanashi <takanashi@gclue.jp>\n\
Architecture: arm64\n\
Depends: \n\
Description: OpenCV version 3.3.1\n"\
> /package_build/opencv-3.3.1/DEBIAN/control \
&& fakeroot dpkg-deb --build opencv-3.3.1

mkdir -p $SCRIPT_DIR/../binary
mv -f /package_build/opencv-3.3.1.deb $SCRIPT_DIR/../binary
