Android Java Inference Interface for TensorFlowをBazelでbuildして、so, jarファイルを生成。Android Studioのプロジェクトに組み込む。

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android

# Android Java Inference Interface for TensorFlow

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android

```shell
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
```

## WORKSPACEの設定

tensorflowフォルダに移動し、WORKSPACEファイルを編集する。

```
$ cd tensorflow
$ ls -l WORKSPACE
```
WORKSPACEの`android_sdk_repository`, `android_ndk_repository`を自分の環境に合わせる。`android_sdk_repository`のapi_levelとbuild_tools_versionをAndroid Studioの環境に合わせる。

WORKSPACE
```
android_sdk_repository(
    name = "androidsdk",
    api_level = 24,
    build_tools_version = "25.0.2",
    # Replace with path to Android SDK on your system
    path = "/Users/sasakiakira/Library/Android/sdk/",
)

android_ndk_repository(
    name="androidndk",
    path="/Users/sasakiakira/Library/Android/sdk/ndk-bundle/",
    api_level=21)
```

## libtensorflow_inference.soのBuild

```shell
$ bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a
```

これで、libtensorflow_inference.so が生成される

```shell
$ ls bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so
```

|ファイル名||
|:--|:--|
|libandroid_tensorflow_lib.lo|core TensorFlow runtime + ops for linking into other libraries|
|libtensorflow_inference.so|core TF runtime+ops with added JNI bindings|


## libandroid_tensorflow_inference_java.jarのBuild

```
$ bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
```

これで、libandroid_tensorflow_inference_java.jarが生成される

```
$ bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar
```

setting.gradle
```
include ':app',':TensorFlow-Android-Inference'
findProject(":TensorFlow-Android-Inference").projectDir =
        new File("/Users/sasakiakira/Documents/workspace_ai_android/android/bazel/tensorflow/tensorflow/contrib/android/cmake")
```

# ヒント

https://github.com/tensorflow/tensorflow/issues/6356

Build.gradle
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/build.gradle

