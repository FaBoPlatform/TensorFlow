# Google Cloud DataLab

Google Cloud DataLabは、データを探索、視覚化、分析、変換するための、生産性の高いインタラクティブな統合ツールです。

本チュートリアルでは、Google Cloud DataLabを使って説明していきます。

Cloud MLでProject IDを取得します。

[CloudMLで実行]
```shell
$ gcloud projects list
```

取得したプロジェクト名と対応付されたDataLabをLocalマシン上に起動します。
下記はOS X用。Windows用のDocker設定は、[https://cloud.google.com/datalab/docs/quickstarts/quickstart-local](https://cloud.google.com/datalab/docs/quickstarts/quickstart-local)を参照。

[OS Xで実行]
```shell
$ cd ~
$ mkdir -p ./datalab
$ docker run -it -p "127.0.0.1:8081:8080" -v "${HOME}/datalab:/content" \
 -e "PROJECT_ID=プロジェクトID"  \
gcr.io/cloud-datalab/datalab:local
```

Browserで、localhost:8081に接続します。

![](/img/datalab001.png)

![](/img/datalab002.png)

![](/img/datalab003.png)

![](/img/datalab004.png)

![](/img/datalab005.png)

![](/img/datalab006.png)

```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()
```