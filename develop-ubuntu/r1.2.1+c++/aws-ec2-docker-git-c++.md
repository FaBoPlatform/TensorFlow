# AWS Ubuntu 16.04 - TensorFlor r1.2.1 CPU/XLA/MKL/CPU最適化 Python/C++  
PythonとC++の両方を使えるようにします  

普段使いのAWS c4.xlargeインスタンスの処理性能をあげるために、TensorFlowビルドに最適化オプションを指定することに。GPUインスタンスでも同様に出来るけれども、速度向上は体感できなかった。しかし、CPUインスタンスでは高速化が出来た。

###### SSD物体認識処理
* c4.8xlarge 36コア 1画像処理
  * CPU最適化なし：time\:0.25347185 clock\:7.83642500
  * CPU最適化あり：time\:0.17184091 clock\:4.64863300
* c4.xlarge 4コア 1画像処理
  * CPU最適化なし：time\:1.50084400 clock\:5.85320900
  * CPU最適化あり：time\:0.54308510 clock\:2.02857800


## CPU最適化
* ビルドに使ったインスタンス
  * c4.8xlarge
* bazel build
  * bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfxsr --copt=-mxsave --copt=-mxsaveopt --copt=-mbmi --copt=-mbmi2 --copt=-mcx16 --copt=-maes --copt=-mmmx --copt=-mabm --copt=-msse --copt=-msse2 --copt=-msse4.2 --copt=-msse4.1 --copt=-mssse3 --copt=-mf16c --copt=-mfma -k --verbose_failures \/\/tensorflow\/tools\/pip_package:build_pip_package

###### 利用可能なオプションの探し方
TesnorFlowのコードをコマンドラインで実行する際にでるAVX/SSE4.1が使えるという情報を元に決めてもいい。CPUが対応しているかどうかは
```
cat /proc/cpuinfo
```
> flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq monitor est ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm fsgsbase bmi1 avx2 smep bmi2 erms invpcid xsaveopt ida

を見て確認する。  
ここからgccで指定できるオプションを指定する。
```
gcc -march=native -Q --help=target | grep enable
```

>-m64                        		[enabled]  
>-m80387                     		[enabled]  
>-m96bit-long-double         		[enabled]  
>-mabm                       		[enabled]  
>-maes                       		[enabled]  
>-malign-stringops           		[enabled]  
>-mavx                       		[enabled]  
>-mavx2                      		[enabled]  
>-mbmi                       		[enabled]  
>-mbmi2                      		[enabled]  
>-mcx16                      		[enabled]  
>-mf16c                      		[enabled]  
>-mfancy-math-387            		[enabled]  
>-mfentry                    		[enabled]  
>-mfma                       		[enabled]  
>-mfp-ret-in-387             		[enabled]  
>-mfsgsbase                  		[enabled]  
>-mfxsr                      		[enabled]  
>-mglibc                     		[enabled]  
>-mhard-float                		[enabled]  
>-mhle                       		[enabled]  
>-mieee-fp                   		[enabled]  
>-mlong-double-80            		[enabled]  
>-mlzcnt                     		[enabled]  
>-mmmx                       		[enabled]  
>-mmovbe                     		[enabled]  
>-mpclmul                    		[enabled]  
>-mpopcnt                    		[enabled]  
>-mpush-args                 		[enabled]  
>-mrdrnd                     		[enabled]  
>-mred-zone                  		[enabled]  
>-mrtm                       		[enabled]  
>-msahf                      		[enabled]  
>-msse                       		[enabled]  
>-msse2                      		[enabled]  
>-msse3                      		[enabled]  
>-msse4                      		[enabled]  
>-msse4.1                    		[enabled]  
>-msse4.2                    		[enabled]  
>-mssse3                     		[enabled]  
>-mstackrealign              		[enabled]  
>-mtls-direct-seg-refs       		[enabled]  
>-mxsave                     		[enabled]  
>-mxsaveopt                  		[enabled]  

###### r1.2からの更新の場合、headerファイルコピースクリプトで
    find: File system loop detected; 'third_party/eigen3/mkl_include/include' is part of the same file system loop as 'third_party/eigen3/mkl_include'.
と出るため、headerファイルコピーする際には
```
rm /tensorflow/third_party/mkl/mklml_lnx_2018.0.20170425/include/include
```
を実行し、原因となっているシンボリックリンクを削除する

###### あとは通常のTensorFlowビルド
Dockerはr.1.1の時のDockerを使っている。  
今回は環境はr1.1参考に。  
更新はビルド関連部分だけ。  
今回ビルドにはc4.8xlarge Intel(R) Xeon(R) CPU E5-2666 v3 @ 2.90GHz 36コアを使っている。  
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

となっていて、c4.xlargeとc4.8xlargeは共にHaswellなのでMKLを利用できる。  
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

cp output/bazel /usr/local/bin/  
```  
## TensorFlow r1.2.1 コンパイル  
r1.1の時のgitディレクトリがある場合はfetch/checkoutで更新する  
無ければgit cloneでコードを取得すること  
```  
cd /tensorflow  
git fetch  
git checkout r1.2  
git pull
```  
CPUインスタンスの場合、  
>Do you wish to build TensorFlow with OpenCL support? [y/N]   
No OpenCL support will be enabled for TensorFlow  
Do you wish to build TensorFlow with CUDA support? [y/N]   
No CUDA support will be enabled for TensorFlow  

OpenCLサポート？の次にCUDAサポート？が来る。CPUなのでCUDA使わないで進む。  
  
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
>Do you wish to build TensorFlow with CUDA support? [y/N]  
>No CUDA support will be enabled for TensorFlow  
>................  
>INFO: Starting clean (this may take a while). Consider using --async if the clean takes more than several minutes.  
>Configuration finished  

```  
time bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfxsr --copt=-mxsave --copt=-mxsaveopt --copt=-mbmi --copt=-mbmi2 --copt=-mcx16 --copt=-maes --copt=-mmmx --copt=-mabm --copt=-msse --copt=-msse2 --copt=-msse4.2 --copt=-msse4.1 --copt=-mssse3 --copt=-mf16c --copt=-mfma -k --verbose_failures //tensorflow/tools/pip_package:build_pip_package
```  

```  
# whlを生成  
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg  
# TensorFlow インストール  
pip install /tmp/tensorflow_pkg/tensorflow-1.2.1-cp27-cp27mu-linux_x86_64.whl  
# .soを作成  
time bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfxsr --copt=-mxsave --copt=-mxsaveopt --copt=-mbmi --copt=-mbmi2 --copt=-mcx16 --copt=-maes --copt=-mmmx --copt=-mabm --copt=-msse --copt=-msse2 --copt=-msse4.2 --copt=-msse4.1 --copt=-mssse3 --copt=-mf16c --copt=-mfma -k --verbose_failures //tensorflow:libtensorflow_cc.so

# /usr/local/lib にインストールする  
install -m 0644 bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/  
ldconfig  
# チュートリアルのコンパイル  
time bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfxsr --copt=-mxsave --copt=-mxsaveopt --copt=-mbmi --copt=-mbmi2 --copt=-mcx16 --copt=-maes --copt=-mmmx --copt=-mabm --copt=-msse --copt=-msse2 --copt=-msse4.2 --copt=-msse4.1 --copt=-mssse3 --copt=-mf16c --copt=-mfma -k --verbose_failures //tensorflow/cc:tutorials_example_trainer

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
