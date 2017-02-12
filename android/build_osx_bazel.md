
## TensorFlow1.0.0をGitからClone

TensorFlow 1.0をGitでCloneする。

```shell
$ git clone -b r1.0 --recurse-submodules https://github.com/tensorflow/tensorflow.git
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

生成したlibtensorflow_inference.so を任意の場所にコピーしておく。


## Bazelでlibandroid_tensorflow_inference_java.jarをBuild

```shell
$ bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
```

libandroid_tensorflow_inference_java.jarが生成されたか確認する。

```shell
$ bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar
```

生成したlibtensorflow_inference.jar を任意の場所にコピーしておく。

## Android Studioのプロジェクトへの取り込み

`libtensorflow_inference.so`と`libtensorflow_inference.jar`をそれぞれ、`libs/armedabi-v7a`と`libs`に新規フォルダを作成してコピーする。

![](/img/android_studio101.png)

また、Modelデータは、`aseets`フォイルにコピーする。

今回は、[モデルデータの保存と読込](../model_basic/tensorflow_model.md)で作成した[model.pb](https://github.com/FaBoPlatform/TensorFlow/raw/master/model/model.pb) を使用する。

また、build.gradleを下記のように書き直し、TensorFlowのライブラリをアプリ内で使用できるようにする。

![](/img/android_studio102.png)

```shell
# Gradle Scripts
# build.gradle(Module:app)
android {
    ...
    sourceSets {
        main {
            jniLibs.srcDirs = ['libs']
            assets.srcDirs = ['assets']
        }
    }
}
...
dependencies {
    ...
    compile files('libs/libandroid_tensorflow_interface_java.jar')
}
```

## Androidアプリ

```Java
package io.fabo.helloandroid;

import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    private final static String TAG = "TF_LOG";

    static {
        System.loadLibrary("tensorflow_inference");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TensorFlowInferenceInterface mTensorFlowIF = new TensorFlowInferenceInterface();
        AssetManager mAssetManager = getAssets();
        int result = mTensorFlowIF.initializeTensorFlow(mAssetManager, "file:///android_asset/model.pb");

        mTensorFlowIF.enableStatLogging(true);
        Log.i(TAG, "result:" + result);

        int[] a_value = new int[1];
        a_value[0] = 3;
        int[] b_value = new int[1];
        b_value[0] = 4;

        mTensorFlowIF.fillNodeInt("input_a",new int[] {1}, a_value);
        mTensorFlowIF.fillNodeInt("input_b",new int[] {1}, b_value);

        int[] result_value = new int[1];
        mTensorFlowIF.runInference(new String[] {"add_op"});
        mTensorFlowIF.readNodeInt("add_op", result_value);

        Log.i(TAG, "result_value:" + result_value[0]);
    }
}
```
## 参考

* TensorFlowモデルの読み込み方法
    https://github.com/FaBoPlatform/TensorFlow/blob/master/android/run.md
* TensorFlowモデル
    https://github.com/FaBoPlatform/TensorFlow/blob/master/android/model.pb

