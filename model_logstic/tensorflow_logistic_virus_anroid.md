# ロジスティック回帰 Androidで動作

## TensorFlow

* [Notebook](https://github.com/FaBoPlatform/TensorFlow/blob/master/model_logstic/virus_android.ipynb)
* [出力したグラフ](https://github.com/FaBoPlatform/TensorFlow/blob/master/model_logstic/graph-virus.pbtxt)
* [出力したグラフ(バイナリ)](https://github.com/FaBoPlatform/TensorFlow/blob/master/model_logstic/graph-virus.pb)

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

predict_op200, app_opも処理が走るが値が変わらない。

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
        x_value[0] = (float) 9.0;
        x_value[1] = (float) 3.0;

        mTensorFlowIF.fillNodeFloat("input",new int[] {2}, x_value);

        // Add
        mTensorFlowIF.runInference(new String[] {"add_op"});
        float[] result_value1 = new float[2];
        mTensorFlowIF.readNodeFloat("add_op", result_value1);
        Log.i(TAG, "result_add:  " + result_value1[0]);
        Log.i(TAG, "result_add:  " + result_value1[1]);

        // Predict
        int x_cols=2;
        int x_rows=10;
        float x_value2[] = {
                /* x_rows[0] */ -2,-2,
                /* x_rows[1] */ 0,0,
                /* x_rows[2] */ -2,2,
                /* x_rows[3] */ -2 /* error? */
                /* x_rows[4] */ /* error? */
                /* x_rows[...] */ /* error? */
        };
        mTensorFlowIF.fillNodeFloat("input",new int[] {x_rows,x_cols}, x_value2);
        int runInference = mTensorFlowIF.runInference(new String[]{"predict_op200"});

        float[] result_value2 = new float[x_rows];
        mTensorFlowIF.readNodeFloat("predict_op200", result_value2);

        // 入力値->出力を整形
        String x[]= new String[x_rows];
        String message = "";
            for (int row = 0; row < x_rows; row++) {
                message = "x(";

                for (int col = 0; col < x_cols; col++) {
                    x[col] = x_value2.length > (row * x_cols + col) ? Float.toString(x_value2[row * x_cols + col]) : "?";
                    if (col > 0) {
                        message += ",";
                    }
                    message += x[col];
                }
                message += ") -> result_predict:  " + result_value2[row];
                Log.i(TAG, message);
            }
        }
    }
}
```

出力
```
I/TF_LOG: result_add:  18.0
I/TF_LOG: result_add:  6.0

I/TF_LOG: x(-2.0,-2.0) -> result_predict:  1.0
I/TF_LOG: x(0.0,0.0) -> result_predict:  0.0
I/TF_LOG: x(-2.0,2.0) -> result_predict:  1.0
I/TF_LOG: x(-2.0,?) -> result_predict:  0.0
I/TF_LOG: x(?,?) -> result_predict:  0.0
I/TF_LOG: x(?,?) -> result_predict:  0.0
I/TF_LOG: x(?,?) -> result_predict:  0.0
I/TF_LOG: x(?,?) -> result_predict:  0.0
I/TF_LOG: x(?,?) -> result_predict:  0.0
I/TF_LOG: x(?,?) -> result_predict:  0.0
```

