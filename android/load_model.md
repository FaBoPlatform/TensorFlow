# モデルデータを読込

Androidアプリでモデルデータを読み込む

## Sample

```java
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

        Log.i(TAG, "result:" + result);
    }
}
```

build.gradle
```
android {
    compileSdkVersion 24
    buildToolsVersion "25.0.2"
    defaultConfig {
        applicationId "io.fabo.helloandroid"
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
            manifest.srcFile 'src/main/AndroidManifest.xml'
            java.srcDirs = ['src/main/java/', '../../contrib/android/java']
            //resources.srcDirs = ['src']
            //aidl.srcDirs = ['src']
            //renderscript.srcDirs = ['src']
            res.srcDirs = ['src/main/res']
            assets.srcDirs = ['asset']
            jniLibs.srcDirs = ['src/main/libs']
        }
    }

}
```

![](/img/android_model01.png)

