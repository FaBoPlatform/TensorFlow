# Tensorflowのグラフ操作 Part3

## 名前空間

TensorFlowでサポートされている名前空間(name scope)を利用することで、ノードの管理を便利にすることが可能になる。

名前空間は例えばフォルダ機能のようなもので、ニューラルネットの「第一層目`first_layer`」の重みノード`w`、ニューラルネットの「第二層目`second_layer`」の重みノード`w`といったように命名規則に一貫性を持たせることができる。

### サンプルコード

```python
# coding:utf-8
import tensorflow as tf
# 名前空間の指定
with tf.name_scope("input_layer"):
    x = tf.placeholder(shape=[64,64], dtype=tf.float32, name="x")
with tf.name_scope("first_layer"):
    w1 = tf.Variable(x, name="weights")
with tf.name_scope("second_layer"):
    # ネストを深くすることが可能
    with tf.name_scope("sub_scope"):
        a = tf.Variable([1.0], name="a")
        b = tf.Variable([2.0], name="b")
    w2 = tf.Variable(w1+a+b, name="weights")
with tf.name_scope("output_layer"):
    y = tf.Variable(w2, name="y")
g = tf.get_default_graph()
# オペレーション一覧を表示する
for op in g.get_operations():
    print op.name
# 可視化
tf.summary.FileWriter('graph_log', graph=g)
```

### 実行結果

```
input_layer/x
first_layer/weights
first_layer/weights/Assign
first_layer/weights/read
second_layer/constant/a
second_layer/constant/b
second_layer/add
second_layer/add_1
second_layer/weights
second_layer/weights/Assign
second_layer/weights/read
output_layer/y
output_layer/y/Assign
output_layer/y/read
```

## TensorBoardによる可視化

`tensorboard --logdir=./graph_log`

![](/img/name_scope_tfboard.jpg)

---

## Tensorを指定して取得する

`graph.get_tensor_by_name(...)`を用いてTensorを取得する。

### サンプルコード

```python
# coding:utf-8
import tensorflow as tf
with tf.name_scope("input_layer"):
    x = tf.placeholder(shape=[64,64], dtype=tf.float32, name="x")
with tf.name_scope("first_layer"):
    w1 = tf.Variable(x, name="weights")
with tf.name_scope("second_layer"):
    with tf.name_scope("sub_scope"):
        a = tf.Variable([1.0], name="a")
        b = tf.Variable([2.0], name="b")
    w2 = tf.Variable(w1+a+b, name="weights")
with tf.name_scope("output_layer"):
    y = tf.Variable(w2, name="y")
g = tf.get_default_graph()
w = g.get_tensor_by_name("first_layer/weights:0")
print(w)
a = g.get_tensor_by_name("second_layer/sub_scope/a:0")
print(a)
```

### 実行結果

```
Tensor("first_layer/weights:0", shape=(64, 64), dtype=float32_ref)
Tensor("second_layer/sub_scope/a:0", shape=(1,), dtype=float32_ref)
```