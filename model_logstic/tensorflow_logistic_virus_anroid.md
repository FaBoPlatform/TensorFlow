# ロジスティック回帰 Androidで動作

## TensorFlow

* [Notebook]("./virus_android.ipynb")
* [出力したグラフ]("./graph-virus.pbtxt")
* [出力したグラフ(バイナリ)]("./graph-virus.pb")

## Androidソースコード

```java
package fabo.io.hellotensorflow;

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
    protected void onCreate (Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TensorFlowInferenceInterface mTensorFlowIF = new TensorFlowInferenceInterface();
        AssetManager mAssetManager = getAssets();
        int result = mTensorFlowIF.initializeTensorFlow(mAssetManager, "file:///android_asset/graph-virus.pb");

        mTensorFlowIF.enableStatLogging(true);
        Log.i(TAG, "---------");
        Log.i(TAG, "initializeTensorFlow:result:" + result);

        float[] x_value = new float[2];
        x_value[0] = (float) 2.0;
        x_value[1] = (float) 2.0;

        mTensorFlowIF.fillNodeFloat("input",new int[] {0,2}, x_value);

        // Add
        mTensorFlowIF.runInference(new String[] {"add_op"});
        float[] result_value1 = new float[2];
        mTensorFlowIF.readNodeFloat("add_op", result_value1);
        Log.i(TAG, "result_add:  " + result_value1[0]);
        Log.i(TAG, "result_add:  " + result_value1[1]);

        // Predict
        mTensorFlowIF.runInference(new String[] {"predict_op200"});
        float[] result_value2 = new float[2];
        mTensorFlowIF.readNodeFloat("predict_op200", result_value2);
        Log.i(TAG, "result_predict:  " + result_value2[0]);
        Log.i(TAG, "result_predict:  " + result_value2[1]);
    }

}
```

predict_op200, ap_opも処理が走るが値が変わらない。

```java
package fabo.io.hellotensorflow;

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
    protected void onCreate (Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TensorFlowInferenceInterface mTensorFlowIF = new TensorFlowInferenceInterface();
        AssetManager mAssetManager = getAssets();
        int result = mTensorFlowIF.initializeTensorFlow(mAssetManager, "file:///android_asset/graph-virus.pb");

        mTensorFlowIF.enableStatLogging(true);
        Log.i(TAG, "---------");
        Log.i(TAG, "initializeTensorFlow:result:" + result);

        float[] x_value = new float[2];
        x_value[0] = (float) 2.0;
        x_value[1] = (float) 2.0;

        mTensorFlowIF.fillNodeFloat("input",new int[] {2}, x_value);

        // Add
        mTensorFlowIF.runInference(new String[] {"add_op"});
        float[] result_value1 = new float[2];
        mTensorFlowIF.readNodeFloat("add_op", result_value1);
        Log.i(TAG, "result_add:  " + result_value1[0]);
        Log.i(TAG, "result_add:  " + result_value1[1]);

        // Predict
        mTensorFlowIF.runInference(new String[] {"predict_op200"});
        float[] result_value2 = new float[2];
        mTensorFlowIF.readNodeFloat("predict_op200", result_value2);
        Log.i(TAG, "result_predict:  " + result_value2[0]);
        Log.i(TAG, "result_predict:  " + result_value2[1]);
    }

}
```

ap_opが値が変わるが、predict_op200でエラー

```
/fabo.io.hellotensorflow E/native: tensorflow_inference_jni.cc:233 Error during inference: Invalid argument: In[0] is not a matrix
                                                                   [[Node: MatMul = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/cpu:0"](_recv_input_0, constant_w)]]
02-13 06:13:56.592 32487-32487/fabo.io.hellotensorflow E/native: tensorflow_inference_jni.cc:170 Output [predict_op200] not found, aborting!
```

