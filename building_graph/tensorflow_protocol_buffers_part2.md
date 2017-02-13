# プロトコルバッファ Part2

TensorFlowのファイル形式はすべてプロトコルバッファがベースとなっている。そのためTensorFlowで生成されたデータはCやPython、その他の言語で容易に扱うことができる。

![](/img/tf-proto-lang.jpg)

## TensorFlowにおけるプロトコルバッファ

`Graph`から`GraphDef`の流れ

1. `Graph`オブジェクトの生成
2. `as_graph_def()`による`GraphDef`オブジェクトの生成

`Graph`オブジェクトはTensorとオペレーションの情報を持つオブジェクトであり、`GraphDef`オブジェクトはプロトコルバッファ用ライブラリによって生成されるオブジェクトである。

## 保存形式

* テキスト
	* 拡張子`.pbtxt`
	* 可読性あり
	* 編集可能
* バイナリ
	* 拡張子`.pb`
	* テキスト形式よりサイズが小さい

## サンプルコード

1+2を行うグラフを構築し、そのプロトコルバッファを確認する。

```python
# coding:utf-8
import tensorflow as tf
a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
print graph_def
```

## 実行結果

プロトコルバッファに似た文字列に出力されていることが分かる。

```
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Add"
  op: "Add"
  input: "Const"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 17
}
```

## 補足:`NodeDef`

グラフにおけるノードの情報を扱う`NodeDef`のプロトコルバッファ

|メンバ|説明|
|:-:|:-:|
|`name`|グラフ上の一意となるノード名|
|`op`|実行するオペレーション|
|`input`|入力されるノードのリスト|
|`device`|実行環境の情報|
|`attr`|ノードのkey/valueデータ|

## 参考

* [A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/how_tos/tool_developers/)

