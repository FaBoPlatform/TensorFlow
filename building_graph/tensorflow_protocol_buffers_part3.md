# プロトコルバッファ Part3
## 学習済みmodelをprotocol bufferファイルに保存するポイント
TensorFlowのプロトコルバッファにはポイントが3つある。
* データ入力ノードとなるplaceholderやdequeue_op、出力ノードとなるpredictionやaccuracyには名前を付けておくこと
* checkpoint形式で保存しておくこと
* tf.VariableはConstに変換してから保存する必要があること

###### ポイント1
入力ノード、出力ノードは後で
input_x= graph.get_tensor_by_name('input_x:0')
output_y= graph.get_tensor_by_name('output_y:0')
のように名前でノードを取得することになるのでわかりやすくするために名前を付けておく。
###### ポイント2
checkpointでの保存は2行で出来る。
```python
saver = tf.train.Saver()
... 学習 ...
saver.save(sess, MODEL_DIR + '/model.ckpt')
```
checkpointでは学習済みモデル情報（graphのメタ情報）と学習値（tf.Variable）が別々のファイルに保存される。メタ情報を読み込んでgraphを復元する際に、学習用に設定したCPU/GPUのデバイス依存を削除することが出来る。これによってポータビリティが向上する。
（ただし、tf.Variableで変数を用意していない学習しないモデルの場合はcheckpointでの保存はできない。その場合、pbへの書き出し、利用にポイント3は不要になる。）
###### ポイント3
通常のモデルは学習済みのWeightやBiasを保持するためのtf.Variableの変数を持つ。この値をpbファイルに保存することは出来ないため、 graph_util.convert_variables_to_constants() を使ってConstに変換したgraphをpbファイルに保存することになる。

![](./images/TensorFlow-model-save.png)
