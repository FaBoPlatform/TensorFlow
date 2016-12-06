
# CloudMLで必要なパッケージを有効にする

# プロジェクトを作成する　

https://console.cloud.google.com/iam-admin/projects?_ga=1.178290596.1433708546.1475329198

# 課金を有効にする

https://console.cloud.google.com/billing

# APIを有効にする

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

# CloudMLの設定

```shell
$ curl https://raw.githubusercontent.com/GoogleCloudPlatform/cloudml-samples/master/tools/setup_cloud_shell.sh | bash
$ export PATH=${HOME}/.local/bin:${PATH}
$ curl https://raw.githubusercontent.com/GoogleCloudPlatform/cloudml-samples/master/tools/check_environment.py | python
$ gcloud beta ml init-project
```

バケットの設定、your_bucket_nameを任意の名前にする

```shell
$ PROJECT_ID=$(gcloud config list project --format "value(core.project)")
$ BUCKET_NAME=${PROJECT_ID}-ml
$ BUCKET_NAME="your_bucket_name"
$ gsutil mb -l us-central1 gs://$BUCKET_NAME
```

# HelloTensor

![](/img/ipython001.png)

```python
$ ipython
```

![](/img/ipython002.png)

```python
import tensorflow as tf

hello = tf.constant('Hello')
sess = tf.Session()
print sess.run(hello)
```

