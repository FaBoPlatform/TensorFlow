########################################
# OpenMPI v3.0.0 パッケージ作成/インストール
########################################
apt-get remove -y openmpi-common libopenmpi1.10
mkdir -p /compile \
&& cd /compile \
&& wget --no-check-certificate https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz \
&& tar -zxvf openmpi-3.0.0.tar.gz \
&& cd openmpi-3.0.0 \
&& ./configure --prefix=/package_build/openmpi-3.0.0/usr --enable-mpi-thread-multiple --with-cuda=/usr/local/cuda/targets/aarch64-linux --with-cuda-libdir=/usr/lib/aarch64-linux-gnu CXXFLAGS='-std=c++11 -march=native' \
&& make install

mkdir -p /package_build/openmpi-3.0.0/DEBIAN \
&& cd /package_build \
&& echo -e "Source: openmpi-3.0.0\n\
Package: openmpi\n\
Version: 3.0.0\n\
Priority: optional\n\
Maintainer: Yoshiroh Takanashi <takanashi@gclue.jp>\n\
Architecture: arm64\n\
Depends: \n\
Description: OpenMPI version 3.0.0\n"\
> /package_build/openmpi-3.0.0/DEBIAN/control \
&& fakeroot dpkg-deb --build openmpi-3.0.0

cd /package_build \
&& dpkg -i openmpi-3.0.0.deb

#dpkg --add-architectureでaarch64を登録しないとパッケージがデフォルトのamd64になってしまい、自作パッケージをインストール出来ない。
