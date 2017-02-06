Android Java Inference Interface for TensorFlowをModuleとして、Android Studioに取り込む。BazzelのBuild等もAndroid Studio内から実行。


https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android/cmake


setting.gradle
```
include ':app',':TensorFlow-Android-Inference'
findProject(":TensorFlow-Android-Inference").projectDir =
        new File("/Users/sasakiakira/Documents/workspace_ai_android/android/bazel/tensorflow/tensorflow/contrib/android/cmake")
```

build.gradle
```
def bazel_location = '/usr/local/bin/bazel'
def cpuType = 'armeabi-v7a'
def nativeDir = 'libs/' + cpuType

project.buildDir = 'gradleBuild'
getProject().setBuildDir('gradleBuild')

apply plugin: 'com.android.application'

android {
    compileSdkVersion 24
    buildToolsVersion "25.0.2"
    defaultConfig {
        applicationId "io.fabo.virus"
        minSdkVersion 21
        targetSdkVersion 24
        versionCode 1
        versionName "1.0"
        testInstrumentationRunner "android.support.test.runner.AndroidJUnitRunner"
    }
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }

    lintOptions {
        abortOnError false
    }

    sourceSets {
        main {
            manifest.srcFile 'AndroidManifest.xml'
            java.srcDirs = ['src', '../../contrib/android/java']
            resources.srcDirs = ['src']
            aidl.srcDirs = ['src']
            renderscript.srcDirs = ['src']
            res.srcDirs = ['res']
            assets.srcDirs = ['assets']
            jniLibs.srcDirs = ['libs']
        }

        debug.setRoot('build-types/debug')
        release.setRoot('build-types/release')
    }
}

dependencies {
    compile fileTree(dir: 'libs', include: ['*.jar'])
    androidTestCompile('com.android.support.test.espresso:espresso-core:2.2.2', {
        exclude group: 'com.android.support', module: 'support-annotations'
    })
    compile 'com.android.support:appcompat-v7:24.2.1'
    testCompile 'junit:junit:4.12'

    debugCompile project(path: ':TensorFlow-Android-Inference', configuration: 'debug')
    releaseCompile project(path: ':TensorFlow-Android-Inference', configuration: 'release')
}

task buildNative(type:Exec) {
    workingDir '/Users/sasakiakira/Documents/workspace_ai_android/android/bazel/tensorflow'
    commandLine bazel_location, 'build', '-c', 'opt', \
      'tensorflow/examples/android:tensorflow_native_libs', \
       '--crosstool_top=//external:android/crosstool', \
       '--cpu=' + cpuType, \
       '--host_crosstool_top=@bazel_tools//tools/cpp:toolchain'
}

task copyNativeLibs(type: Copy) {
    from('/Users/sasakiakira/Documents/workspace_ai_android/android/bazel/tensorflow/bazel-bin/tensorflow/contrib/android/') { include '**/*.so' }
    into nativeDir
    duplicatesStrategy = 'include'
}

copyNativeLibs.dependsOn buildNative
assemble.dependsOn copyNativeLibs
task findbugs(type: FindBugs, dependsOn: 'assembleDebug') {
    copyNativeLibs
}
```



```
$ brew instal automake
$ brew install libtool
```
