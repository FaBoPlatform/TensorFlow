# AWS Ubuntu 16.04 - TensorFlor r1.2 GPU/XLA/MKL Python/C++  
PythonとC++の両方を使えるようにします  

Dockerはr.1.1の時のDockerを使っている。  
今回は環境はr1.1参考に。  
更新はビルド関連部分だけ。  
今回ビルドにはp2.8xlarge Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz 32コアを使っている。  
MKL対応CPUは、  
https://github.com/01org/mkl-dnn  
>Intel MKL-DNN supports Intel(R) 64 architecture processors and is optimized for  
>  
    Intel Atom(R) processor with Intel(R) SSE4.1 support  
    4th, 5th, 6th and 7th generation Intel(R) Core processor  
    Intel(R) Xeon(R) processor E5 v3 family (code named Haswell)  
    Intel(R) Xeon(R) processor E5 v4 family (code named Broadwell)  
    Intel(R) Xeon(R) Platinum processor family (code name Skylake)  
    Intel(R) Xeon Phi(TM) product family x200 (code named Knights Landing)  
    Future Intel(R) Xeon Phi(TM) processor (code named Knights Mill)  

となっていて、p2.xlargeとp2.8xlargeは共にBroadwellなのでMKLを利用できる。  
作業はすべてDockerコンテナ内で行う  

```  
apt-get update  
apt-get dist-upgrade  
```  
* Dockerコンテナ再起動  
  
Intel Math Kernel Library  
MKLを有効にすると、configureでlocateを使ったファイルパス検索が実行される。Dockerにlocateが無いので、apt-get install locateでインストールするが、locateはupdatedbを実行してDBに記録されたファイルパスを検索するため、updatedbの実行も必要になる  
TensorFlowのconfigure時にMKLMLをダウンロードするため、IntelからMKLをダウンロード/インストールしておく必要はない。これたぶん別物。  
  
```  
apt-get install locate cpio  
updatedb  
```  
* updatedbでlocateのファイルパスDBを更新する  
bazelはビルドし直す  

```  
########################################  
# bazel-0.5.1 ビルド  
########################################  
cd /bazel  
wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.5.1/bazel-0.5.1-dist.zip  
unzip bazel-0.5.1-dist.zip -d bazel-0.5.1  
cd bazel-0.5.1  
time ./compile.sh  
real	2m31.561s  
user	19m59.060s  
sys	2m32.900s  
cp output/bazel /usr/local/bin/  
```  
## TensorFlow r1.2 コンパイル  
r1.1の時のgitディレクトリがある場合はfetch/checkoutで更新する  
無ければgit cloneでコードを取得すること  
```  
cd /tensorflow  
git fetch  
git checkout r1.2  
```  
CPUインスタンスの場合、  
>Do you wish to build TensorFlow with OpenCL support? [y/N]   
No OpenCL support will be enabled for TensorFlow  
Do you wish to build TensorFlow with CUDA support? [y/N]   
No CUDA support will be enabled for TensorFlow  

OpenCLサポート？の次にCUDAサポート？が来る。CPUなのでCUDA使わないで進む。  
ところが、GPUインスタンスの場合、  
>No OpenCL support will be enabled for TensorFlow  
>Do you want to use clang as CUDA compiler? [y/N]   
>nvcc will be used as CUDA compiler  

OpenCLサポート？の次に、CUDAコンパイラにclang使う？と来ている。CUDA使うこと前提で話が進んでいるので、ここでYを選択しないように注意！clangは十分なテストがされていない。  
  
```  
./configure  
```  
>Please specify the location of python. [Default is /usr/bin/python]:   
>Found possible Python library paths:  
>  /usr/local/lib/python2.7/dist-packages  
>  /usr/lib/python2.7/dist-packages  
>Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]  
>  
>Using python library path: /usr/local/lib/python2.7/dist-packages  
>Do you wish to build TensorFlow with MKL support? [y/N] Y  
>MKL support will be enabled for TensorFlow  
>Do you wish to download MKL LIB from the web? [Y/n]   
>Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:   
>Do you wish to use jemalloc as the malloc implementation? [Y/n]   
>jemalloc enabled  
>Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]   
>No Google Cloud Platform support will be enabled for TensorFlow  
>Do you wish to build TensorFlow with Hadoop File System support? [y/N]   
>No Hadoop File System support will be enabled for TensorFlow  
>Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] Y  
>XLA JIT support will be enabled for TensorFlow  
>Do you wish to build TensorFlow with VERBS support? [y/N]   
>No VERBS support will be enabled for TensorFlow  
>Do you wish to build TensorFlow with OpenCL support? [y/N]   
>No OpenCL support will be enabled for TensorFlow  
>Do you want to use clang as CUDA compiler? [y/N]   
>nvcc will be used as CUDA compiler  
>Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to use system default]:   
>Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:   
>Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:   
>Please specify the cuDNN version you want to use. [Leave empty to use system default]:   
>Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:   
>Extracting Bazel installation...  
>.  
>INFO: Starting clean (this may take a while). Consider using --async if the clean takes more than several minutes.  
>Configuration finished  

32コアのp2.8xlargeはTensorFlowコンパイルが早い！ 

```  
time bazel build -c opt --output_filter='^//tensorflow' --config=cuda --verbose_failures //tensorflow/tools/pip_package:build_pip_package  
```  
>real	20m21.467s  
>user	0m0.044s  
>sys	0m0.144s  

```  
# whlを生成  
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg  
# TensorFlow インストール  
pip install /tmp/tensorflow_pkg/tensorflow-1.2.0-cp27-cp27mu-linux_x86_64.whl  
# .soを作成  
time bazel build -c opt --config=cuda --verbose_failures //tensorflow:libtensorflow_cc.so  
real	0m44.869s  
user	0m0.004s  
sys	0m0.052s  
# /usr/local/lib にインストールする  
install -m 0644 bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/  
ldconfig  
# チュートリアルのコンパイル  
time bazel build -c opt //tensorflow/cc:tutorials_example_trainer  
real	4m12.963s  
user	0m0.012s  
sys	0m0.068s  
# チュートリアルの実行  
./bazel-bin/tensorflow/cc/tutorials_example_trainer  
```  
  
###### header ファイルをインストールする
vi ./tensorflow_header_install.sh  
```  
#!/bin/bash -eu
# -*- coding: utf-8 -*-
# http://memo.saitodev.com/home/tensorflow/build/
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

GPUを使うかどうかは、bazel buildオプションの --config=cuda があるかどうかだけ。  
configureも注意が必要だけども。