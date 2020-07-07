########################################
# TensorFlow r1.4.1
########################################
SCRIPT_DIR=$(cd $(dirname $0); pwd)
# TensorFlow r1.4.1 download

#Cuda compilation tools, release 9.0, V9.0.252
#9.0.252

mkdir -p /compile \
&& cd /compile \
&& git clone -b r1.4 https://github.com/tensorflow/tensorflow

cat /usr/local/cuda/version.txt
nvcc --version
grep "#define CUDNN_MAJOR" -A2 /usr/include/cudnn.h
#TF_CUDA_VERSION=8.0 (lib 8.0.84, nvcc 8.0.72)
#TF_CUDNN_VERSION=6.0.21

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
    TF_NEED_OPENCL=0 \
    TF_NEED_GDR=0 \
    TF_ENABLE_XLA=0 \
    TF_NEED_VERBS=0 \
    TF_NEED_CUDA=1 \
    TF_CUDA_CLANG=0 \
    TF_CUDA_COMPUTE_CAPABILITIES=6.2 \
    TF_CUDA_VERSION=8.0 \
    TF_CUDNN_VERSION=`grep "#define CUDNN_MAJOR" -A2 /usr/include/cudnn.h | sed -e 's/\#define\s[A-Z_]*\s*\(.*\)$/\1/g' | sed -e ':loop; N; $!b loop; s/\n/./g'` \
    TF_NEED_MPI=0 \
    ./configure

# TX2は、メモリ消費量を抑えたいのでOpenMPIは使わない
# TX2は、GCP,HDFS,S3のどれかが原因で実行時にエラーとなる。この3つを切るとr1.4.1が出来上がる
# TX2は、XLAを無効にする。理由はJetPack3.2/TensorFlow r1.4.1では公式Object Detectionがエラーになってしまう。

# IntelじゃないのでMKLは使わない
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/pip_package:build_pip_package \
&& bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

mkdir -p $SCRIPT_DIR/../binary
mv -f /tmp/tensorflow_pkg/tensorflow-1.4.1-cp36-cp36m-linux_aarch64.whl $SCRIPT_DIR/../binary


# Inspecting Graphs
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/graph_transforms:summarize_graph

# Optimizing for Deployment
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/graph_transforms:transform_graph

# memmapped
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/contrib/util:convert_graphdef_memmapped_format

# Quantize Graph
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/quantization:quantize_graph


# benchmark
cd /compile/tensorflow/tools
git clone https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/benchmark
cd /compile/tensorflow
time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/tools/benchmark:benchmark_model

cd /compile/tensorflow/bazel-bin/tensorflow/tools/benchmark
wget --no-check-certificate https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
unzip inception5h.zip

/compile/tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"
