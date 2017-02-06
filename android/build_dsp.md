AndroidのDSPに最適化されたBuildをおこなう。

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android/cmake

Dependencies

TensorFlow-Android-Inference depends on the TensorFlow static libs already built in your local TensorFlow repo directory. For Linux/Mac OS, build_all_android.sh is used in build.gradle to build it. It DOES take time to build the core libs; so, by default, it is commented out to avoid confusion (otherwise Android Studio would appear to hang during opening the project). To enable it, refer to the comment in

build.gradle

DSP向けのBuildもできる模様

```
$ brew instal automake
$ brew install libtool
```