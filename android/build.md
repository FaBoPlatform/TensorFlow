Androidアプリ開発環境でTensorFlowの学習済みグラフ(プロトコルバッファ形式のモデル)を読み込み実行するまで

<img src="diagram/build-diagram1.png" alt="TensorFlow Android Interface library build environment" title="TensorFlow Android Interface library build environment">

```
########################################
# Java8 インストール
########################################
add-apt-repository "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main"
apt-get update
add-apt-repository ppa:webupd8team/y-ppa-manager
apt-get install oracle-java8-installer
```
```
########################################
# Android SDK インストール
########################################
# https://developer.android.com/studio/index.html
# コマンドラインツール(tools_r25.2.3-linux.zip)
# Android StudioとTensorFlow Android Interfaceのビルド設定に合わせて環境変数を設定する
export ${ANDROID_HOME}=$HOME/Android/Sdk
export ${NDK_ROOT}=${ANDROID_HOME}/ndk-bundle
mkdir -p ${ANDROID_HOME}

unzip tools_r25.2.3-linux.zip
mv tools ${ANDROID_HOME}
${ANDROID_HOME}tools/bin/sdkmanager --update

# TensorFlow Android Interfaceのビルド設定に合わせてbuild-toolsとndk-bundleとtoolsをインストールする
${ANDROID_HOME}/tools/bin/sdkmanager --list
${ANDROID_HOME}/tools/bin/sdkmanager \
"build-tools;24.0.2" \
ndk-bundle \
tools \
"platforms;android-24" \
"platforms;android-21" \
"cmake3.6.3155560" \
"patcher;v4" \
"extras;android;m2repository"

# ワーニングが出る時はファイルを作る
vi ~/.android/repositories.cfg
### User Sources for Android SDK Manager
count=0
```
```
########################################
# build package インストール
########################################
# TensorFlow Android Interface ライブラリ作成用
apt-get install git automake libtool zlib1g-dev
```

# Android Java Inference Interface for TensorFlow

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android

ライブラリが欲しいのですが、ソースコードからbuildしないと手に入らないみたいなのでbuildします。ここではbazelではなく、build_all_android.shを使ってビルドする方法を記載します。
<img src="diagram/build-diagram2.png" alt="TensorFlow Android Interface library build" title="TensorFlow Android Interface library build">

```
########################################
# TensorFlow r1.0 source code
########################################
mkdir ~/github; cd ~/github
git clone -b r1.0 --recurse-submodules https://github.com/tensorflow/tensorflow.git
```
```
########################################
# TensorFlow Android Interface build with build_all_android.sh
########################################
#export NDK_ROOT=${ANDROID_HOME}/ndk-bundle
#${ANDROID_HOME}/tools/bin/sdkmanager "platforms;android-24" "platforms;android-21" "platforms;android-21" "build-tools;24.0.2"
# -march=nativeを削除する r0.12ではなかったけど、r1.0.0.rc2から-march=nativeが追加されていてエラーになるので削除する
# https://github.com/tensorflow/tensorflow/commit/c4e3d4a74e86fce3a09badd20952f067ff340f32
# tensorflow/tensorflow/contrib/makefile/Makefile L:140
vi ${HOME}/github/tensorflow/tensorflow/contrib/makefile/Makefile
#OPTFLAGS := -O2 -march=native
OPTFLAGS := -O2

${HOME}/github/tensorflow/tensorflow/contrib/makefile/build_all_android.sh
エラーになったらpackage不足かTensorFlowの開発が進んでMakefileやコードが変わっているためだと思うので、エラー内容をみて修正する必要があります。
```

## Android アプリを作成する
<img src="diagram/build-diagram3.png" alt="Android App with TensorFlow Android Interface" title="Android App with TensorFlow Android Interface">
```
########################################
# Android Studioをインストールする
########################################
# https://developer.android.com/studio/index.html?hl=ja
unzip android-studio-ide-145.3537739-linux.zip
mv android-studio ~/
```
```
########################################
# Android Studio - Hello Application JNIプロジェクトを作成する
########################################
# Gradle Scripts
# settings.gradle (Project Settings)
include ':app',':TensorFlow-Android-Inference'
findProject(":TensorFlow-Android-Inference").projectDir =
        new File("/home/guppy/github/tensorflow/tensorflow/contrib/android/cmake")


# Gradle Scripts
# build.gradle(Module:app)
# tensorflow_inferenceではなく、TensorFlow-Android-Inferenceとする。
dependencies {
    ...
    debugCompile project(path: ':TensorFlow-Android-Inference', configuration: 'debug')
    releaseCompile project(path: ':TensorFlow-Android-Inference', configuration: 'release')
}
```
## TensorFlowモデルの読み込み方法
https://github.com/FaBoPlatform/TensorFlow/blob/master/android/run.md
## TensorFlowモデル
https://github.com/FaBoPlatform/TensorFlow/blob/master/android/model.pb

