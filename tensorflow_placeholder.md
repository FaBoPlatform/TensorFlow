# Constant

Constantは、定数である。

> tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)


|引数名|概要|
|:--|:--|
| value | dtypeで指定したOutputの定数の値 |
| dtype | Tensorの型 |
| shape | Tensorの形状, 無指定の場合は任意の形状のTensorを渡せる |
| name | 名前(Const) |
| verify_shape | 値の形状のVarification(検証)をするか |

## Sample

![](/img/placeholder01.png)


# Variable
Variableは、変数である。

> tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)

|引数名|概要|
|:--|:--|
| initial_value | 初期値 |
| trainable| |
| collections| |
| validate_shape| |
| caching_device| |
| name | 名前(Const) |
| variable_def| |
| dtype | Tensorの型 |
| expected_shape |  |
| import_scope| | 

## Sample

![](/img/placeholder02.png)


# Placeholder

> tf.placeholder(dtype, shape=None, name=None)


|引数名|概要|
|:--|:--|
| dtype | Tensorの型 |
| shape | Tensorの形状, 無指定の場合は任意の形状のTensorを渡せる |
| name | 操作の名前 |

## Sample

![](/img/placeholder03.png)

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/placeholder.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/placeholder.ipynb)

## 参考

[Placeholder](https://www.tensorflow.org/api_docs/python/io_ops/placeholders#placeholder)

