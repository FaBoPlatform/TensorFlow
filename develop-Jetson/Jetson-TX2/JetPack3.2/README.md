# Jetson TX2 環境構築

## Jetson TX2 JetPack 3.2 環境構築方法
> \# After JetPack 3.2 installation  
> \# ssh login to TX2  
> ssh ubuntu@192.168.x.x  
> 
> sudo su  
> cd setup_scripts  
> ./setup.sh  
> \#wait reboot  
> 
> sudo su  
> cd install_scripts  
> ./install.sh  
> \#wait hours  

## 内容
 * #### 初期設定 (setup_scripts/setup.sh)
   * CPUファン自動起動 (setup_cpufun.sh)
   * Ubuntu 16.04 パッケージ更新 (setup_update.sh)
   * .bashrc書き換え (setup_bash.sh)
   * .dircolors追加 (setup_dircolors.sh)
   * reboot
 * #### TensorFlow r1.6.0 インストール (install_scripts/install.sh)
   * Ubuntu 16.04 パッケージ更新
   * Python 3.6.3 インストール (install_python3.6.sh)
   * pip3 インストール (install_pip3.sh)
   * jupyter インストール (install_jupyter.sh)
   * Java8 インストール (install_java8.sh)
   * Build Tools インストール (install_build_tools.sh)
   * CUDA deviceQuery ビルド (install_cuda_deviceQuery.sh)
   * OpenCV用にCUDAヘッダーパッチ適用 (cv_patch.sh)
   * OpenCV 3.4.1 インストール (install_opencv-3.4.1.sh)
   * bazel 0.10.0 ビルド (build_bazel-0.10.0.sh)
   * TensorFlow r1.6.0 インストール (install_tensorflow-r1.6.0.sh)
 * #### パッケージ作成
   * OpenCV 3.4.1 パッケージ作成 (build_opencv-3.4.1.sh)
   * OpenMPI 3.4.1 パッケージ作成 (build_openmpi-3.4.1.sh)
   * TensorFlow r1.6.0 pipパッケージ作成 (build_tensorflow-r1.6.0.sh)

## Jupyter 起動方法
install_scripts/install_jupyter.shでTX2起動時に自動起動するように設定してある。  
初期パスワードはmypassword  
> #/etc/init.d/jupyterd
> env PASSWORD=mypassword jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000

## Jupyter アクセス方法
ブラウザでアクセスする。  
パスワードは起動時に環境変数に指定したmypassword  
> http://IPアドレス:8888/


## 議論
 * TX2: メモリが豊富ではないので学習には向かない
 * TX2: GPUしか使わないのでSWAPは要らない
 * TensorFlow: メモリ消費抑制、エラー回避のためにJEMALLOC,CUDAのみ有効
 * TensorFlow: MKLはIntelなのでARMのTX2では使わない
 * TX2: DenverコアはOpenCVビルドに失敗するので使わない
 * TX2: パッケージはARM64で作成する
 * TensorFlow: XLAを無効 <- 有効だとJetPack 3.2ではObject Detectionのobject_detection_tutorial.ipynbをJupyterで実行するとThe kernel appears to have died. It will restart automatically.で落ちる。無効だと実行できる。
 * TX2: JetPack 3.1はCUDA 8.0.84, nvcc 8.0.72でバージョンが違う
 * TX2: JetPack 3.2はCUDA 9.0.252, nvcc 9.0.252でバージョンが一致
 * TX2: ハードウェア関連はJetPack 3.1が安定
 * TX2: JetPack 3.2はまだ新しいためカスタムボード用カーネル非対応