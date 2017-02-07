# Versionを表示する

AndroidアプリでJNIでBridgeされているTensorFlowのVersionを表示する

## Sample

```java
package io.fabo.helloandroid;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import org.tensorflow.TensorFlow;

public class MainActivity extends AppCompatActivity {

    private final static String TAG = "TF_LOG";

    static {
        System.loadLibrary("tensorflow_inference");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String version = TensorFlow.version();
        Log.i(TAG, "version" + version);
    }
}
```

![](/img/android_version01.png)
