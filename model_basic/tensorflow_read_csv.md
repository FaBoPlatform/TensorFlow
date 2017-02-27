# CSV形式のファイルを読み込む

TensorFlowプログラムにデータを投入する主な方法は以下の通り。

1. 各ステップを実行する度に独自のPythonコードでデータを与える
1. ファイルからデータを読み込む
1. 小さいデータセットの場合、`tf.constant`と`tf.Variable`にデータを持たせる

[参考](https://www.tensorflow.org/programmers_guide/reading_data)

本記事では**2. ファイルからデータを読み込む**をTensorFlow内で扱う方法を紹介する。

## サンプルコード

CSV形式のファイル(`hoge.csv`、`foo.csv`、`bar.csv`)から各列の値を読み込む。

構築されるTensorFlowグラフ：

![](/img/graph_read_csv.jpg)

```python
#!/usr/bin/env python
# coding:utf-8
"""
file name: read_csv.py
Reading CSV format data
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


### CSV形式ファイルからデータを読み込む処理をグラフとして構築する
# 読み込むファイルのリスト
filenames = ["./hoge.csv","./foo.csv","./bar.csv"]
filename_queue = tf.train.string_input_producer(filenames, name="input_producer")
# テキストを1行読み込む
reader = tf.TextLineReader(name="textline_reader")
key, value = reader.read(filename_queue, name="reader_read")

record_defaults = [[1], [1], [1], [1], [1]]  # 型の代表値
# 文字列"1,1,1,1,1"を5つの整数値としてパースする
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults, name="decode_csv")
# 最初の4つの整数値を特徴量ベクトル、最後の値をラベルとしてまとめる
features = tf.stack([col1, col2, col3, col4], name="features")
label = tf.stack([col5], name="label")


sess = tf.Session()
coord = tf.train.Coordinator()  # スレッドを管理するクラス
# `sess.run(...)`を実行する前に`tf.train.start_queue_runners(...)`を実行する
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for _ in range(20):
    _features_, _label = sess.run([features, label])
    print(_features_, _label)

coord.request_stop()
coord.join(threads)
sess.close()
```

## 実行結果

`hoge.csv`、`foo.csv`、`bar.csv`を作成

```bash
$ python -c 'import sys;import numpy as np;np.savetxt("%s.csv"%sys.argv[1],np.random.randint(0,100,(5,5)),delimiter=",",fmt="%i")' hoge
$ python -c 'import sys;import numpy as np;np.savetxt("%s.csv"%sys.argv[1],np.random.randint(0,100,(5,5)),delimiter=",",fmt="%i")' foo
$ python -c 'import sys;import numpy as np;np.savetxt("%s.csv"%sys.argv[1],np.random.randint(0,100,(5,5)),delimiter=",",fmt="%i")' bar
```

`python read_csv.py` :

```
[54 60 97  9] [93]
[71 16 40  3] [69]
[59 76 78 80] [31]
[28 75 53 53] [63]
[50 93 86 38] [58]
[51 58 67 63] [27]
[72 10 99 84] [85]
[35  4 18 66] [98]
[83 71 90 63] [36]
[54 12 52 11] [73]
[80 71 97 96] [45]
[11  0  3 25] [51]
[ 8  6 81 57] [93]
[86 81 83 53] [62]
[60 59  4 98] [76]
[80 71 97 96] [45]
[11  0  3 25] [51]
[ 8  6 81 57] [93]
[86 81 83 53] [62]
[60 59  4 98] [76]
```

## 実行環境

* Python 3.6.0
* TensorFlow 1.0.0

## 参考

* https://www.tensorflow.org/programmers_guide/reading_data
* https://www.tensorflow.org/api_docs/python/tf/train/start_queue_runners
* https://www.tensorflow.org/api_docs/python/tf/decode_csv
* https://www.tensorflow.org/versions/r0.11/api_docs/python/train/coordinator_and_queuerunner