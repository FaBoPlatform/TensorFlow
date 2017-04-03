# モデルデータを読込 その2

Androidアプリでモデルデータを読み込む  
今回はヤマメの判定を行う学習モデルを作成し、そのモデルデータを読み込んでテストを行う
データセットについては以下から入手する

```
$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data
```

## 学習モデルのプログラム

```python
# coding:utf-8
# tensorflow version1.0.0
import numpy as np
import tensorflow as tf

### データの準備
# データセットの読み込み
dataset = np.genfromtxt("./bezdekIris.data", delimiter=',', dtype=[float, float, float, float, "S32"])
# データセットの順序をランダムに並べ替える
np.random.shuffle(dataset)

def get_labels(dataset):
    """ラベル(正解データ)を1ofKベクトルに変換する"""
    raw_labels = [item[4] for item in dataset]
    labels = []
    for l in raw_labels:
        if l == "Iris-setosa":
            labels.append([1.0,0.0,0.0])
        elif l == "Iris-versicolor":
            labels.append([0.0,1.0,0.0])
        elif l == "Iris-virginica":
            labels.append([0.0,0.0,1.0])
    return np.array(labels)

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[:4] for item in dataset]
    return np.array(raw_data)

# ラベル
labels = get_labels(dataset)
# データ
data = get_data(dataset)
# 訓練データとテストデータに分割する
# 訓練用データ
train_labels = labels[:120]
train_data = data[:120]
# テスト用データ
test_labels = labels[120:]
test_data = data[120:]


### モデルをTensor形式で実装

# ラベルを格納するPlaceholder
t = tf.placeholder(tf.float32, shape=(None,3))
# 入力データを格納するPlaceholder nameをつけておく
X = tf.placeholder(tf.float32, shape=(None,4),name="input")

# 隠れ層のノード数
node_num = 1024
w_hidden = tf.Variable(tf.truncated_normal([4,node_num]))
b_hidden = tf.Variable(tf.zeros([node_num]))
f_hidden = tf.matmul(X, w_hidden) + b_hidden
hidden_layer = tf.nn.relu(f_hidden)

# 出力層
w_output = tf.Variable(tf.zeros([node_num,3]))
b_output = tf.Variable(tf.zeros([3]))
f_output = tf.matmul(hidden_layer, w_output) + b_output
#出力結果の値が入る nameをつけておく
p = tf.nn.softmax(f_output,name="output")

# 交差エントロピー
cross_entropy = t * tf.log(p)
# 誤差関数
loss = -tf.reduce_mean(cross_entropy)
# トレーニングアルゴリズム
# 勾配降下法 学習率0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)
# モデルの予測と正解が一致しているか調べる
correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
# モデルの精度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
run_metadata = tf.RunMetadata()
### 学習の実行
with tf.Session() as sess:
    #ディレクトリは事前に作成しておく
    ckpt = tf.train.get_checkpoint_state('./ckpt-iris')
    if ckpt:
      # checkpointファイルから最後に保存したモデルへのパスを取得する
      last_model = ckpt.model_checkpoint_path
      print("load {0}".format(last_model))
      # 学習済みモデルを読み込む
      saver.restore(sess, last_model)
    else:
      print("initialization")
    # ログの設定
    tf.summary.histogram("Hidden_layer_wights", w_hidden)
    tf.summary.histogram("Hidden_layer_biases", b_hidden)
    tf.summary.histogram("Output_layer_wights", w_output)
    tf.summary.histogram("Output_layer_wights", b_output)
    tf.summary.scalar("Accuracy", accuracy)
    tf.summary.scalar("Loss", loss)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./iris_cassification_log", sess.graph)
    #初期化
    sess.run(tf.global_variables_initializer())

    i = 0
    for _ in range(5000):
        i += 1
        # トレーニング
        sess.run(train_step, feed_dict={X:train_data,
                                        t:train_labels})
        # 200ステップごとに精度を出力
        if i % 200 == 0:
            # コストと精度を出力
            train_summary,train_loss, train_acc = sess.run([summary,loss,accuracy], feed_dict={X:train_data,t:train_labels})
            writer.add_summary(train_summary,i)
            print "Step: %d" % i
            print "[Train] cost: %f, acc: %f" % (train_loss, train_acc)

    saver.save(sess, "iris-model")
sess.close()

```

## モデルデータ(pbファイル)を作成する

```python
# coding:utf-8
# tensorflow version1.0.0
import tensorflow as tf
from tensorflow.python.framework import graph_util

#モデルデータを作成する
def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    output_graph = model_folder + "/frozen_model.pb"
    print(output_graph)

    output_node_names = "output,input"
    #学習時に計算にcpuやgpuの指定を行なっていた時、読み込む側でその指定に依存しないようにする
    clear_devices = True
    #グラフをインポートする
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
      #保存されている重みやバイアスの変数を復元する
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )
        #モデルデータを書き出す
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
```

## モデルデータをAndroidで読み込みテストする
```java
import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
//導入については開発環境構築(Android)を参照
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    //判定結果を表示するテキストビュー
    private TextView ansView;
    //入力データを入れるテキストフォーム
    private EditText editIrisFeature1;
    private EditText editIrisFeature2;
    private EditText editIrisFeature3;
    private EditText editIrisFeature4;
    //判定を開始するボタン
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
        //ボタンを押した時のイベント
        detectButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
              　//入力データを格納する配列で判定に使う
                float[] features = new float[4];
                //入力のテキストフォームから得る値を取得する
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
                  //未入力のまま開始した時エラーダイアログを表示するようにする
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
                  //判定を開始する
                    onDetectClicked(features);
                }

            }
        });


    }
    private void onDetectClicked(float[] f) {
        TensorFlowInferenceInterface mTensorFlowIF = new TensorFlowInferenceInterface();
        AssetManager mAssetManager = getAssets();
        //モデルデータ読み込み
        int result = mTensorFlowIF.initializeTensorFlow(mAssetManager, "file:///android_asset/iris_FIFO_frozen_model.pb");

        ansView.setText("");
        //入力データを入れる
        //第一引数にはモデル作成時に指定した入力データを格納するPlaceholderのnameを指定する
        //第二引数にはshapeを指定する　今回は一つのサンプルデータを渡してテストするので(1,4)
        //第三引数は入力データ
        mTensorFlowIF.fillNodeFloat("input:0",new int[] {1,4},f);
        //判定結果を格納する配列
        float[] result_value = new float[3];
        //判定を行うこの時にモデル作成時の出力のnameを指定する
        mTensorFlowIF.runInference(new String[] {"output:0"});
        //result_valueに結果を代入する
        mTensorFlowIF.readNodeFloat("output:0", result_value);

        //result_valueには[a,b,c]と値が入っており、一番大きな値が入ってる配列のインデックスが入力データのクラスとなる
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
    //一番大きな値が入っているインデックスを返す
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

データセットからサンプルを選び、正解ラベルが返ってきたら成功
![load_model2_result](/img/load_model2_result.jpg)
