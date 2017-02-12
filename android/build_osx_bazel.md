
## TensorFlow1.0.0をGitからClone

TensorFlow 1.0をGitでCloneする。

```shell
git clone -b r1.0 --recurse-submodules https://github.com/tensorflow/tensorflow.git
```

## BREWのインストール

```shell
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

## Java 8.xのインストール

```shell
$ brew cask install java
```

## Bazelのインストール

```shell
$ brew install bazel
$ brew update bazel
```

## WORKSPACEの編集

|パス|Defaultの場所|
|:--|:--|
|ANDROID_HOME|/Users/username/Library/Android/sdk/|
|NDK_ROOT|${ANDROID_HOME}/ndk-bundle|


tensorflow/WORKSPACE を編集する。android_sdk_repositoryとandroid_ndk_repositoryを自分の環境に合わせて設定する。android sdkのフォルダの下にndk-bundleというフォルダがない場合は、Android StudioのManager for Android SDK and ToolsのAndroid ToolsからNDKをインストールしておく。

![](/img/android_build01.png)

![](/img/android_build02.png)

`tensorflow/WORKSPACE`
```shell
# Uncomment and update the paths in these entries to build the Android demo.
android_sdk_repository(
    name = "androidsdk",
    api_level = 23,
    build_tools_version = "23.0.1",
    # Replace with path to Android SDK on your system
    path = "/Users/sasakiakira/Library/Android/sdk/",
)

android_ndk_repository(
    name="androidndk",
    path="/Users/sasakiakira/Library/Android/sdk/ndk-bundle/",
    api_level=21)
```

## Bazelでlibtensorflow_inference.soをBuild

```shell
$ bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a
```

libtensorflow_inference.soが生成されたか確認する。

```shell
$ ls bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so
```

## Bazelでlibandroid_tensorflow_inference_java.jarをBuild

```shell
$ bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
```

libandroid_tensorflow_inference_java.jarが生成されたか確認する。

```shell
$ bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar
```


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

