# プロトコルバッファ Part4

#### tf.Variableで定義したv1で、v1=v1+1を実行するモデルを作成し、10回実行後checkpointに保存する
```python
# -*- coding: utf-8 -*-
# sample_pb1.py
# v1(tf.Variable)の値を更新してcheckpointファイルに保存する
import tensorflow as tf
import os

# jupyter実行用にGraphを初期化する
tf.reset_default_graph()

# checkpoint保存先ディレクトリ準備
MODEL_DIR = "./model_data"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# graph定義
# ポイント1:重要な要素には判りやすい名前を付けておく
v1 = tf.Variable(initial_value=0.0, name="this_is_my_v1")
v1_add = v1.assign_add(1) # v1=v1+1をv1_addとして用意する

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        _v1 = sess.run(v1_add) # v1=v1+1を実行する
        print("i:%d v1=%d" % (i,_v1))

    # ポイント2:graphと学習済みv1をcheckpointに保存する
    saver.save(sess, MODEL_DIR + '/model.ckpt')
```
実行結果
```
i:0 v1=1
i:1 v1=2
i:2 v1=3
i:3 v1=4
i:4 v1=5
i:5 v1=6
i:6 v1=7
i:7 v1=8
i:8 v1=9
i:9 v1=10
```

#### checkpointを読み込んでpbファイルに保存する
```python
# -*- coding: utf-8 -*-
# sample_pb2.py
# checkpointからmetaを読み込みgraphを復元
# operationを表示
# restoreでv1の値を復元
# tf.VariableをConstに変換したgraphをpbに出力する

import tensorflow as tf
from tensorflow.python.framework import graph_util

# jupyter実行用にGraphを初期化する
tf.reset_default_graph()

# checkpoint保存先ディレクトリ
MODEL_DIR = "./model_data"
# 保存先のpbファイル名
FROZEN_MODEL_NAME="frozen_model.pb"
# デバイス情報を削除する
CLEAR_DEVICES=True
# pbに書き出すoperation名
OUTPUT_NODE_NAMES="this_is_my_v1"

# graphのoperationを表示する
def print_graph_operations(graph):
    # print operations
    print "----- operations in graph -----"
    for op in graph.get_operations():
        print op.name,op.outputs

# graph_defのnodeを表示する
def print_graph_nodes(graph_def):
    # print nodes
    print "----- nodes in graph_def -----"
    for node in graph_def.node:
        print(node)

# checkpointファイルの確認を行う
checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
if not checkpoint:
    # checkpointファイルが見つからない
    print("cannot find checkpoint.")
else:
    # checkpointファイルから最後に保存したモデルへのパスを取得する
    last_model = checkpoint.model_checkpoint_path
    print(("load {0}".format(last_model)))

    # pbファイル名を設定する
    absolute_model_dir = "/".join(last_model.split('/')[:-1])
    frozen_model = absolute_model_dir + "/" + FROZEN_MODEL_NAME

    # checkpointのmetaファイルからGraphを読み込む
    saver = tf.train.import_meta_graph(last_model + '.meta', clear_devices=CLEAR_DEVICES)

    # graph定義を取得する
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()

    # print operations
    print_graph_operations(graph)

    # print nodes
    #print_graph_nodes(graph_def)

    with tf.Session() as sess:
        # 学習済みモデルの値を復元する
        saver.restore(sess, last_model)

        # tf.VariableをConstに変換したgraphを取得する
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            OUTPUT_NODE_NAMES.split(",")
        )

        # pbファイルに保存
        tf.train.write_graph(output_graph_def, MODEL_DIR,
                             FROZEN_MODEL_NAME, as_text=False)
```
実行結果：OUTPUT_NODE_NAMESがわからない時は表示されたoperationからアタリをつける
```
load ./model_data/model.ckpt
----- operations in graph -----
this_is_my_v1/initial_value [<tf.Tensor 'this_is_my_v1/initial_value:0' shape=() dtype=float32>]
this_is_my_v1 [<tf.Tensor 'this_is_my_v1:0' shape=() dtype=float32_ref>]
this_is_my_v1/Assign [<tf.Tensor 'this_is_my_v1/Assign:0' shape=() dtype=float32_ref>]
this_is_my_v1/read [<tf.Tensor 'this_is_my_v1/read:0' shape=() dtype=float32>]
AssignAdd/value [<tf.Tensor 'AssignAdd/value:0' shape=() dtype=float32>]
AssignAdd [<tf.Tensor 'AssignAdd:0' shape=() dtype=float32_ref>]
save/Const [<tf.Tensor 'save/Const:0' shape=() dtype=string>]
save/SaveV2/tensor_names [<tf.Tensor 'save/SaveV2/tensor_names:0' shape=(1,) dtype=string>]
save/SaveV2/shape_and_slices [<tf.Tensor 'save/SaveV2/shape_and_slices:0' shape=(1,) dtype=string>]
save/SaveV2 []
save/control_dependency [<tf.Tensor 'save/control_dependency:0' shape=() dtype=string>]
save/RestoreV2/tensor_names [<tf.Tensor 'save/RestoreV2/tensor_names:0' shape=(1,) dtype=string>]
save/RestoreV2/shape_and_slices [<tf.Tensor 'save/RestoreV2/shape_and_slices:0' shape=(1,) dtype=string>]
save/RestoreV2 [<tf.Tensor 'save/RestoreV2:0' shape=<unknown> dtype=float32>]
save/Assign [<tf.Tensor 'save/Assign:0' shape=() dtype=float32_ref>]
save/restore_all []
init []
INFO:tensorflow:Froze 1 variables.
Converted 1 variables to const ops.
```
#### pbファイルを読み込み、v1を表示する
```python
# -*- coding: utf-8 -*-
# sample_pb3.py
# frozen_model.pbファイルを読み込む
# v1の値を表示する
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# checkpoint保存先ディレクトリ
MODEL_DIR = "./model_data"

# 保存先のpbファイル名
FROZEN_MODEL_NAME="frozen_model.pb"

def print_graph_operations(graph):
    # print operations
    print "----- operations in graph -----"
    for op in graph.get_operations():
        print op.name,op.outputs

def print_graph_nodes(graph_def):
    # print nodes
    print "----- nodes in graph_def -----"
    for node in graph_def.node:
        print(node)

def load_graph(frozen_graph_filename):
    # pbファイルを読み込みgraph定義を復元する
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # pbファイルから復元したgraph_defをカレントgraph_defに設定する
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="my_prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

graph = load_graph(MODEL_DIR+"/"+FROZEN_MODEL_NAME)
graph_def = graph.as_graph_def()

# print operations
print_graph_operations(graph)

v1 = graph.get_tensor_by_name('my_prefix/this_is_my_v1:0') # v1のoperationの出力nodeを取得する

with tf.Session(graph=graph) as sess:
    _v1 = sess.run(v1) # v1の値を取得する
    print("v1=%d" % (_v1))
```
実行結果：my_prefixをつけているため、v1のnodeは'my_prefix/this_is_my_v1:0'となる。(:0はoperationのreturn配列の0番目の配列を意味する。例えばv1_add_opのreturnが return a,bである場合にbを取りたいときは:1となる。ここではv1=tf.Variableであったもの(今はConst)なので:0しかない)
```
----- operations in graph -----
my_prefix/this_is_my_v1 [<tf.Tensor 'my_prefix/this_is_my_v1:0' shape=() dtype=float32>]
v1=10
```