# 再起動が入るsetup.shを先に実行しておく
# apt-get install でエラーが出たら、/usr/bin/pythonのリンクを2.7にしてapt-get removeする
# ModuleNotFoundError: No module named 'ConfigParser'
# sudo rm /usr/bin/python
# sudo ln -s /usr/bin/python2.7 /usr/bin/python

########################################
# Ubuntu 16.04 パッケージ更新
########################################
apt-get update
time apt-get dist-upgrade -y
apt-get install -y htop
apt autoremove -y


########################################
# Python3.6 インストール
########################################
./install_python3.6.sh


########################################
# pip3 install
########################################
./install_pip3.sh


########################################
# Jupyter インストール
########################################
./install_jupyter.sh
# パスワード：mypassword


########################################
# Java8 インストール
########################################
./install_java8.sh


########################################
# Build Tools インストール
########################################
./install_build_tools.sh


########################################
# CUDA deviceQueryビルド
########################################
./install_cuda_deviceQuery.sh


########################################
# パッケージ作成/インストール準備
########################################
# 自作パッケージはTX2デフォルトのarm64で作る
dpkg --print-architecture
#dpkg --print-foreign-architectures
#dpkg --add-architecture aarch64 <- NG. Use arm64 only.
#dpkg --print-foreign-architectures


########################################
# OpenCV用にCUDAヘッダーパッチ適用
########################################
# OpenCV make中にエラーが発生するため、cudaヘッダを書き換えるパッチを当てる
./cv_patch.sh


########################################
# OpenCV 3.4.1 パッケージ作成/インストール
########################################
#./build_opencv-3.4.1.sh
./install_opencv-3.4.1.sh


########################################
# bazel-0.10.0
########################################
./build_bazel-0.10.0.sh


########################################
# OpenMPI v3.0.0 パッケージ作成/インストール
########################################
# Jetson TX2はメモリ消費量を抑えるために不要な機能は無効にしておく
#./build_openmpi-3.0.0.sh
#./install_openmpi-3.0.0.sh


########################################
# TensorFlow r1.6.0
########################################
#./build_tensorflow-r1.6.0.sh
./install_tensorflow-r1.6.0.sh


########################################
# TensorFlow c++ r1.4.1
########################################
#./build_tensorflow-cpp-r1.4.1.sh
#./install_tensorflow-cpp-r1.4.1.sh

