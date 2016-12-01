
# CloudMLで必要なパッケージを有効にする

CloudMLで必要なコンポーネント一覧

* Google Cloud Machine Learning
* Google Dataflow API
* Google Compute Engine API
* Stackdriver Logging API
* Google Cloud Storage
* Google Cloud Storage JSON API
* BigQuery API

有効にするためのリンク

https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,dataflow,compute_component,logging,storage_component,storage_api,bigquery

# ConsoleでCloudMLを表示

https://console.cloud.google.com/ml

# Google Cloud Shellを有効にする

![](/img/ml001.png)

![](/img/ml002.png)

# TensorFlow

VIMで下記を作成して動作を確認する。

```python
import tensorflow as tf

hello = tf.constant('Hello')
sess = tf.Session()
print sess.run(hello)
```

