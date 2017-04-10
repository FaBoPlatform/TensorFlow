# モデルデータを読込 tensorflow1.0.1

Androidアプリでモデルデータを読み込む  
Tensorflowのversionが1.0.1だとAPIの仕様が変更されているのでその2の記事の方は動かない  
以下がversion1.0.1仕様のMainActivity  
```java
package com.example.yamikachan.irisdetector;

import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    private TextView ansView;
    private EditText editIrisFeature1;
    private EditText editIrisFeature2;
    private EditText editIrisFeature3;
    private EditText editIrisFeature4;
    private Button detectButton;
    static {
        System.loadLibrary("tensorflow_inference");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ansView = (TextView) findViewById(R.id.answer_text);

        editIrisFeature1 = (EditText) findViewById(R.id.edit_Iris_feature1);
        editIrisFeature2 = (EditText) findViewById(R.id.edit_Iris_feature2);
        editIrisFeature3 = (EditText) findViewById(R.id.edit_Iris_feature3);
        editIrisFeature4 = (EditText) findViewById(R.id.edit_Iris_feature4);

        detectButton = (Button) findViewById(R.id.detect_Button);
        detectButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                float[] features = new float[4];
                String value1 = null;
                String value2 = null;
                String value3 = null;
                String value4 = null;
                Boolean flag = false;
                try{
                    value1 = editIrisFeature1.getText().toString();
                    value2 = editIrisFeature2.getText().toString();
                    value3 = editIrisFeature3.getText().toString();
                    value4 = editIrisFeature4.getText().toString();

                    features[0] = Float.valueOf(value1);
                    features[1] = Float.valueOf(value2);
                    features[2] = Float.valueOf(value3);
                    features[3] = Float.valueOf(value4);
                }catch (java.lang.NumberFormatException e){
                    flag = true;
                    ansView.setText("");
                    AlertDialog.Builder alertDialog = new AlertDialog.Builder(MainActivity.this);

                    alertDialog.setTitle("input error");
                    alertDialog.setMessage("Please input values");
                    alertDialog.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog,int which) {
                        }
                    });

                    alertDialog.create();
                    alertDialog.show();
                }

                if(!flag){
                    onDetectClicked(features);
                }

            }
        });


    }
    private void onDetectClicked(float[] f) {
        AssetManager mAssetManager = getAssets();
        TensorFlowInferenceInterface mTensorFlowIF = new TensorFlowInferenceInterface(mAssetManager, "file:///android_asset/frozen_model.pb");

        ansView.setText("");

        //入力データを入れる
        //第一引数にはモデル作成時に指定した入力データを格納するPlaceholderのnameを指定する
        //第二引数には入力データ
        //第三引数にはshapeを指定する　今回は一つのサンプルデータを渡してテストするので(1,4)
        mTensorFlowIF.feed("input:0",f,new long[] {1,4});
        //判定結果を格納する配列
        float[] result_value = new float[3];
        //判定を行うこの時にモデル作成時の出力のnameを指定する
        mTensorFlowIF.run(new String[] {"output:0"});
        //result_valueに結果を代入する
        mTensorFlowIF.fetch("output:0", result_value);

        int ansIndex = getAnswer(result_value);
        switch (ansIndex){
            case 0:
                ansView.setText("Detected : Iris-Setosa");
                break;

            case 1:
                ansView.setText("Detected : Iris-versicolor");
                break;

            case 2:
                ansView.setText("Detected : Iris-virginica");
                break;
        }
    }

    private int getAnswer(float[] f){
        int argmax = 0;
        float max = f[0];
        for(int i=0;i<f.length;i++){
            if(max < f[i]){
                max = f[i];
                argmax = i;
            }
        }

        return argmax;
    }
}

```

### activity_main.xml
```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools"
    android:id="@+id/activity_main"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:paddingBottom="@dimen/activity_vertical_margin"
    android:paddingLeft="@dimen/activity_horizontal_margin"
    android:paddingRight="@dimen/activity_horizontal_margin"
    android:paddingTop="@dimen/activity_vertical_margin"
    tools:context="各自のproject名">

    <LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
        android:orientation="horizontal"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:paddingTop="16dp">
        <EditText
            android:id="@+id/edit_Iris_feature1"
            android:inputType="numberDecimal"
            android:layout_width="80dp"
            android:layout_height="wrap_content"
            android:background="#ffffff"
            android:layout_marginLeft="5dp"
            android:layout_marginRight="5dp" />
        <EditText
            android:id="@+id/edit_Iris_feature2"
            android:inputType="numberDecimal"
            android:layout_width="80dp"
            android:layout_height="wrap_content"
            android:background="#ffffff"
            android:layout_marginLeft="5dp"
            android:layout_marginRight="5dp" />
        <EditText
            android:id="@+id/edit_Iris_feature3"
            android:inputType="numberDecimal"
            android:layout_width="80dp"
            android:layout_height="wrap_content"
            android:background="#ffffff"
            android:layout_marginLeft="5dp"
            android:layout_marginRight="5dp" />
        <EditText
            android:id="@+id/edit_Iris_feature4"
            android:inputType="numberDecimal"
            android:layout_width="80dp"
            android:layout_height="wrap_content"
            android:background="#ffffff"
            android:layout_marginLeft="5dp"
            android:layout_marginRight="5dp" />
    </LinearLayout>

    <Button
        android:id="@+id/detect_Button"
        android:text="Button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_centerHorizontal="true"
        android:layout_marginTop="70dp" />

    <TextView
        android:id="@+id/answer_text"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_below="@+id/detect_Button"
        android:textSize="30sp"
        android:layout_margin="30dp"
        android:gravity="center" />

</RelativeLayout>
```

build.gradle
androidの所に下記を追加しておく
```
sourceSets {
    main {
        jniLibs.srcDirs = ['libs']
        assets.srcDirs = ['assets']
    }
}
```

<img src="/img/load_model2_result.png" width="320px" align="left"><br>
データセットからサンプルを選び、正解ラベルが返ってきたら成功
