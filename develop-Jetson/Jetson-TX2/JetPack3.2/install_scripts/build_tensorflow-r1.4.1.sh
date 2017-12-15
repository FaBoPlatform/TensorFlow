########################################
# TensorFlow r1.4.1
########################################
# TensorFlow r1.4.1 download
cat /usr/local/cuda/version.txt
grep "#define CUDNN_MAJOR" -A2 /usr/include/cudnn.h

mkdir -p /compile \
&& cd /compile \
&& git clone -b r1.4 https://github.com/tensorflow/tensorflow

# build/install Python API
cd /compile/tensorflow
env CI_BUILD_PYTHON=python \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc \
    PYTHON_BIN_PATH=/usr/bin/python \
    PYTHON_LIB_PATH=/usr/local/lib/python3.6/dist-packages \
    CC_OPT_FLAGS='-march=native' \
    TF_NEED_JEMALLOC=1 \
    TF_NEED_GCP=0 \
    TF_NEED_CUDA=1 \
    TF_NEED_HDFS=0 \
    TF_NEED_S3=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_GDR=0 \
    TF_ENABLE_XLA=1 \
    TF_NEED_VERBS=0 \
    TF_NEED_CUDA=1 \
    TF_CUDA_CLANG=0 \
    TF_CUDA_COMPUTE_CAPABILITIES=6.2 \
    TF_CUDA_VERSION=9.0.252 \
    TF_CUDNN_VERSION=7.0.5 \
    TF_NEED_MPI=0 \
    ./configure

# TX2は、メモリ消費量を抑えたいのでOpenMPIは使わない
# TX2は、GCP,HDFS,S3のどれかが原因で実行時にエラーとなる。この3つを切るとr1.4.1が出来上がる

# IntelじゃないのでMKLは使わない
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/pip_package:build_pip_package \
&& bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
&& pip3 install --upgrade /tmp/tensorflow_pkg/tensorflow-1.4.1-cp36-cp36m-linux_aarch64.whl

cd /compile
python -c 'import tensorflow as tf; print(tf.__version__)'

rm -rf /var/lib/apt/lists/* && updatedb
