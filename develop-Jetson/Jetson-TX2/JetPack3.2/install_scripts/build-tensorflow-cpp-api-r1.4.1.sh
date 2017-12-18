#!/bin/bash -eu
# -*- coding: utf-8 -*-
# usage:
# { echo "$TENSORFLOW_VERSION";echo "$ARCH";cat /dev/stdin; } |./install-cpp-api.sh
# { echo "1.4.1";echo "amd64";cat /dev/stdin; } |./install-cpp-api.sh
# { echo "1.4.1";echo "arm64";cat /dev/stdin; } |./install-cpp-api.sh
# { echo "1.4.1";echo "aarch64";cat /dev/stdin; } |./install-cpp-api.sh
# 
# http://memo.saitodev.com/home/tensorflow/build/
# https://gist.github.com/saitodev/3cde48806a32272962899693700d9669
# tensorflowを利用するC++ソースをコンパイルする際に、このディレクトリをインクルードパスに追加する。
# コンパイル方法
# g++ a.cpp -std=c++11 -I/usr/local/tensorflow/include -L. -ltensorflow_cc -Wl,-rpath=.


read -p "Input tensorflow verson: " TENSORFLOW_VERSION
PACKAGE_NAME=tensorflow-cpp
read -p "Input Archi(amd64/arm64/aarch64): " ARCHI
HEADER_DIR=/package_build/${PACKAGE_NAME}-${TENSORFLOW_VERSION}/usr/local/tensorflow/include

if [ -e $HEADER_DIR ];
then
    rm -rf $HEADER_DIR
fi

mkdir -p $HEADER_DIR

find tensorflow/core -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
find tensorflow/cc   -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
find tensorflow/c    -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;

find third_party/eigen3 -follow -type f -exec cp --parents {} $HEADER_DIR \;

pushd bazel-genfiles
find tensorflow -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
popd

pushd bazel-tensorflow/external/protobuf_archive/src
find google -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
popd

pushd bazel-tensorflow/external/eigen_archive
find Eigen       -follow -type f -exec cp --parents {} $HEADER_DIR \;
find unsupported -follow -type f -exec cp --parents {} $HEADER_DIR \;
popd

########################################
# パッケージ作成/インストール
########################################
chmod 644  bazel-bin/tensorflow/libtensorflow_cc.so
mkdir -p /package_build/${PACKAGE_NAME}-${TENSORFLOW_VERSION}/usr/local/lib
cp bazel-bin/tensorflow/libtensorflow_cc.so /package_build/${PACKAGE_NAME}-${TENSORFLOW_VERSION}/usr/local/lib

mkdir -p /package_build/${PACKAGE_NAME}-${TENSORFLOW_VERSION}/DEBIAN
cd /package_build
echo "Source: $PACKAGE_NAME
Package: $PACKAGE_NAME
Version: $TENSORFLOW_VERSION
Priority: optional
Maintainer: Yoshiroh Takanashi <takanashi@gclue.jp>
Architecture: $ARCHI
Depends: 
Description: TensorFlow CPP API version $TENSORFLOW_VERSION" \
> /package_build/${PACKAGE_NAME}-${TENSORFLOW_VERSION}/DEBIAN/control \
&& fakeroot dpkg-deb --build ${PACKAGE_NAME}-${TENSORFLOW_VERSION}

echo "${PACKAGE_NAME}-${TENSORFLOW_VERSION}.deb"
