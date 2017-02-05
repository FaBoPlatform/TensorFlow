# CSVファイルの読み込み・保存

CSV(コンマ区切り`,`)およびTSV(タブ区切り`\t`)から値を読み込む。

## Sample

```python
# coding:utf-8
import numpy as np

# CSVファイルの読み込み
x = np.loadtxt('hoge.csv', delimiter=',')
print x
# TSVファイルの読み込み
y = np.loadtxt('piyo.tsv', delimiter='\t')
print y

# CSVファイルの保存
np.savetxt('foo.csv', x, delimiter=',')
```

Datalabでは、現在、Hello World.ipynbを編集中なら、~/datalab/datalab/docsの中に作成する。

`hoge.csv` : 

```
0.2,3.398,592
1.3,0.83,8.8
```

tsvのサンプルはコピーだと、Tabが無効になるので、Tabを自分でうつようにする

`piyo.tsv` : 

```
0.2	3.398	592
1.3	0.83	8.8
```

出力結果

```shell
[[ 1.29809056  1.81463907]
 [ 0.73759474 -0.5688174 ]
 [-1.45300168  0.2555659 ]
 [-0.18510103 -0.18096903]
 [ 0.37652474  0.46395745]
 [-0.45630759 -0.41358571]
 [-1.55313916 -0.09374661]
 [ 1.27759706  2.21715869]
 [ 0.08562266  0.67993018]
 [ 0.14558376  0.37880808]]
```

