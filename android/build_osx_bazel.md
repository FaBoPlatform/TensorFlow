
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


```
apply plugin: 'com.android.library'

android {
    compileSdkVersion 24
    buildToolsVersion "24.0.2"

    // for debugging native code purpose
    publishNonDefault true

    defaultConfig {
        archivesBaseName = "Tensorflow-Android-Inference"
        minSdkVersion 21
        targetSdkVersion 21
        versionCode 1
        versionName "1.0"
        ndk {
            abiFilters  'armeabi-v7a'
        }
        externalNativeBuild {
            cmake {
                arguments '-DANDROID_TOOLCHAIN=gcc',
                          '-DANDROID_STL=gnustl_static'
            }
        }
    }
    sourceSets {
        main {
            java.srcDirs =  ["../java"]
        }
    }

    externalNativeBuild {
        cmake {
            path 'CMakeLists.txt'
        }
    }
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'),
                          'proguard-rules.pro'
        }
    }
}

// Build libtensorflow-core.a if necessary
// Note: the environment needs to be set up already
//    [ such as installing autoconfig, make, etc ]
// How to use:
//    1) install all of the necessary tools to build libtensorflow-core.a
//    2) inside Android Studio IDE, uncomment buildTensorFlow in
//       whenTaskAdded{...}
//    3) re-sync and re-build. It could take a long time if NOT building
//       with multiple processes.
import org.apache.tools.ant.taskdefs.condition.Os

Properties properties = new Properties()
properties.load(project.rootProject.file('local.properties')
                .newDataInputStream())
def ndkDir = properties.getProperty('ndk.dir')
if (ndkDir == null || ndkDir == "") {
    ndkDir = System.getenv('ANDROID_NDK_HOME')
}

if(! Os.isFamily(Os.FAMILY_WINDOWS)) {
    // This script is for non-Windows OS. For Windows OS, MANUALLY build
    // (or copy the built) libs/headers to the
    //    ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen
    // refer to CMakeLists.txt about lib and header directories for details
    task buildTensorflow(type: Exec) {
        group 'buildTensorflowLib'
        workingDir getProjectDir().toString() + '/../../../../'
        environment PATH: '/opt/local/bin:/opt/local/sbin:' +
                          System.getenv('PATH')
        environment NDK_ROOT: ndkDir
        commandLine 'tensorflow/contrib/makefile/build_all_android.sh'
    }

    tasks.whenTaskAdded { task ->
        group 'buildTensorflowLib'
        if (task.name.toLowerCase().contains('sources')) {
            def tensorflowTarget = new File(getProjectDir().toString() +
                    '/../../makefile/gen/lib/libtensorflow-core.a')
            if (!tensorflowTarget.exists()) {
                // Note:
                //    just uncomment this line to use it:
                //    it can take long time to build by default
                //    it is disabled to avoid false first impression
                // task.dependsOn buildTensorflow
            }
        }
    }
}

dependencies {
    compile fileTree(dir: 'libs', include: ['*.jar'])
}

```
