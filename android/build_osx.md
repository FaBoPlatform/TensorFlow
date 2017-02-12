
## TensorFlow1.0.0をGitからClone

TensorFlow 1.0をGitでCloneする。

```shell
git clone -b r1.0 --recurse-submodules https://github.com/tensorflow/tensorflow.git
```

## Android SDKのパスを通す

|パス|Defaultの場所|
|:--|:--|
|ANDROID_HOME|/Users/username/Library/Android/sdk/|
|NDK_ROOT|${ANDROID_HOME}/ndk-bundle|

```shell
$ export ANDROID_HOME=/Users/username/Library/Android/sdk/
$ export NDK_ROOT=${ANDROID_HOME}/ndk-bundle
```

## 必要なパッケージをBREWでインストール

|必要なパッケージ|インストールコマンド|
|:--|:--|
|autoconf|$ brew install autoconf|
|alcocal| $ brew install automake|
|libtool| $ brew install libtool|


```shell
$ cd /tensorflow/tensorflow/contrib/makefile/
$ build_all_android.sh`
```

`-march=native`があると、エラーが発生すえるので、`-march=native`を削除しておく。

`/tensorflow/tensorflow/contrib/makefile/Makefile`
```shell
#OPTFLAGS := -O2 -march=native
$ OPTFLAGS := -O2
```

##  Android Studio - Hello Application JNIプロジェクトを作成する


```shell
# Gradle Scripts
# settings.gradle (Project Settings)
include ':app',':TensorFlow-Android-Inference'
findProject(":TensorFlow-Android-Inference").projectDir =
        new File("/home/guppy/github/tensorflow/tensorflow/contrib/android/cmake")
```

```shell
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

