########################################
# TensorFlow C++ r1.4.1
########################################
cp build-tensorflow-cpp-api-r1.4.1.sh /compile/tensorflow/build-tensorflow-cpp-api.sh


cd /compile/tensorflow \
&& bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow:libtensorflow_cc.so

chmod 755 ./build-tensorflow-cpp-api.sh && sync \
rm -rf third_party/mkl/include/include
{ echo "1.4.1";echo "arm64";cat /dev/stdin; } |./build-tensorflow-cpp-api.sh


cd /compile/tensorflow \
&& time bazel build --config=cuda --config="opt" --copt='-march=native' --copt="-O3" --verbose_failures --subcommands //tensorflow/cc:tutorials_example_trainer

./bazel-bin/tensorflow/cc/tutorials_example_trainer

rm -rf /var/lib/apt/lists/* && updatedb
