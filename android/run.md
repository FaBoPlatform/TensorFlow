# 学習済みモデルを実行する

それでは、[モデルデータの保存と読み込み](http://docs.fabo.io/tensorflow/model_basic/tensorflow_model.html)で作成した学習済みデータを実際にAndroidアプリ内から実行します。

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
