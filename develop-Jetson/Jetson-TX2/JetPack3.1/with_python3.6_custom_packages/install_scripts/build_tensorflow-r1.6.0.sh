########################################
# TensorFlow r1.6.0
########################################
#real	260m59.500s
#user	2m32.980s
#sys	0m30.664s
SCRIPT_DIR=$(cd $(dirname $0); pwd)
# TensorFlow r1.6.0 download

#Cuda compilation tools, release 9.0, V9.0.252
#9.0.252

mkdir -p /compile \
&& cd /compile \
&& git clone -b r1.6 https://github.com/tensorflow/tensorflow

# CUDA,CUDNNバージョン確認
cat /usr/local/cuda/version.txt
nvcc --version
grep "#define CUDNN_MAJOR" -A2 /usr/include/cudnn.h
#TF_CUDA_VERSION=8.0 (lib 8.0.84, nvcc 8.0.72)
#TF_CUDNN_VERSION=6.0.21
# CUDAはx.xまでを指定可能
# CUDNNはx.x.xまで指定可能
# find ./ -type f | xargs grep -nH -A 5 -B 5 TensorRT 
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
    TF_NEED_HDFS=0 \
    TF_NEED_S3=0 \
    TF_ENABLE_XLA=0 \
    TF_NEED_GDR=0 \
    TF_NEED_VERBS=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_OPENCL_SYCL=0 \
    TF_SET_ANDROID_WORKSPACE=0 \
    TF_NEED_CUDA=1 \
    TF_CUDA_CLANG=0 \
    TF_CUDA_COMPUTE_CAPABILITIES=6.2 \
    TF_CUDA_VERSION=8.0 \
    TF_CUDNN_VERSION=`grep "#define CUDNN_MAJOR" -A2 /usr/include/cudnn.h | sed -e 's/\#define\s[A-Z_]*\s*\(.*\)$/\1/g' | sed -e ':loop; N; $!b loop; s/\n/./g'` \
    TF_NEED_MPI=0 \
    TF_NEED_KAFKA=0 \
    TF_NEED_TENSORRT=0 \
    ./configure

# TX2は、メモリ消費量を抑えたいのでOpenMPI,GCP,HDFS,S3,XLAを無効にする
# XLAを無効 <- 有効だとJetPack 3.2ではObject Detectionのobject_detection_tutorial.ipynbをJupyterで実行するとThe kernel appears to have died. It will restart automatically.で落ちる。無効だと実行できる。

# IntelじゃないのでMKLは使わない
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/pip_package:build_pip_package \
&& bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# 150 min

mkdir -p $SCRIPT_DIR/../binary
mv -f /tmp/tensorflow_pkg/tensorflow-1.6.0-cp36-cp36m-linux_aarch64.whl $SCRIPT_DIR/../binary


# benchmark
cd /compile/tensorflow/tools
git clone https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/benchmark
cd /compile/tensorflow
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/benchmark:benchmark_model
#real	0m50.780s
#user	0m0.024s
#sys	0m0.020s


# Inspecting Graphs
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/graph_transforms:summarize_graph

# Optimizing for Deployment
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/graph_transforms:transform_graph

# memmapped
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/contrib/util:convert_graphdef_memmapped_format

# Quantize Graph
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/quantization:quantize_graph


cd /compile/tensorflow/bazel-bin/tensorflow/tools/benchmark
wget --no-check-certificate https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
unzip inception5h.zip

/compile/tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"
