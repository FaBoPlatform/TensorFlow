# Tensorflowのグラフ操作 Part1

TensorFlowにおけるグラフ(Graph)は`tf.Operation`のオブジェクトの集合である。また、tf.OperationはTensorを入力および出力に持つノードとなっている。

## サンプルコード 1

グラフが構築される様子を確認する。

* `tf.get_default_graph()`により、TensorFlowで使われるデフォルトのグラフにアクセスする
* グラフのOperation一覧を取得するには、`graph.get_operations()`を使用する

```python
#!/usr/bin/env python
# coding:utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

# グラフの取得
graph = tf.get_default_graph()
# ノード一覧
print("*** STEP1 ***", graph.get_operations())
x = tf.placeholder(tf.float32, shape=[2,2], name="matrix_x")  # 2x2 Tensor
print("*** STEP2 ***", graph.get_operations())
y = tf.placeholder(tf.float32, shape=[2,2], name="matrix_y")  # 2x2 Tensor
print("*** STEP3 ***", graph.get_operations())
matmul = tf.matmul(x, y, name="matrix_mul")  # xとyの乗算
print("*** STEP4 ***", graph.get_operations())
```

実行結果 :

Operationノードがグラフに追加されていることが分かる。

```
*** STEP1 *** []
*** STEP2 *** [<tensorflow.python.framework.ops.Operation object at 0x10d325850>]
*** STEP3 *** [<tensorflow.python.framework.ops.Operation object at 0x10d325850>, <tensorflow.python.framework.ops.Operation object at 0x10d325bd0>]
*** STEP4 *** [<tensorflow.python.framework.ops.Operation object at 0x10d325850>, <tensorflow.python.framework.ops.Operation object at 0x10d325bd0>, <tensorflow.python.framework.ops.Operation object at 0x10d325c50>]
```

## サンプルコード 2

グラフ内のノード情報を取得する。

* `op.name` : ノードの名前
* `op.type` : ノードの型
* `op.op_def` : ノードのProtocol buffer

```
#!/usr/bin/env python
# coding:utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

# グラフの取得
graph = tf.get_default_graph()

# ノード一覧
x = tf.placeholder(tf.float32, shape=[2,2], name="matrix_x")  # 2x2 Tensor
y = tf.placeholder(tf.float32, shape=[2,2], name="matrix_y")  # 2x2 Tensor
matmul = tf.matmul(x, y, name="matrix_mul")  # xとyの乗算

for op in graph.get_operations():
    # ノードの名前
    print("***name***", op.name)
    # ノードの型
    print("***type***", op.type)
    # ノードの情報を表すprotocol buffer
    print("***op_def***", op.op_def)
```

実行結果 :

```
***name*** matrix_x
***type*** Placeholder
***op_def*** name: "Placeholder"
output_arg {
  name: "output"
  type_attr: "dtype"
}
attr {
  name: "dtype"
  type: "type"
}
attr {
  name: "shape"
  type: "shape"
  default_value {
    shape {
    }
  }
}

***name*** matrix_y
***type*** Placeholder
***op_def*** name: "Placeholder"
output_arg {
  name: "output"
  type_attr: "dtype"
}
attr {
  name: "dtype"
  type: "type"
}
attr {
  name: "shape"
  type: "shape"
  default_value {
    shape {
    }
  }
}

***name*** matrix_mul
***type*** MatMul
***op_def*** name: "MatMul"
input_arg {
  name: "a"
  type_attr: "T"
}
input_arg {
  name: "b"
  type_attr: "T"
}
output_arg {
  name: "product"
  type_attr: "T"
}
attr {
  name: "transpose_a"
  type: "bool"
  default_value {
    b: false
  }
}
attr {
  name: "transpose_b"
  type: "bool"
  default_value {
    b: false
  }
}
attr {
  name: "T"
  type: "type"
  allowed_values {
    list {
      type: DT_HALF
      type: DT_FLOAT
      type: DT_DOUBLE
      type: DT_INT32
      type: DT_COMPLEX64
      type: DT_COMPLEX128
    }
  }
}
```
