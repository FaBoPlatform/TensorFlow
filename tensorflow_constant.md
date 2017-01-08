
# Constant

> tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)


|引数名|概要|
|:--|:--|
| value | dtypeで指定したOutputの定数の値 |
| dtype | Tensorの型 |
| shape | Tensorの形状, 無指定の場合は任意の形状のTensorを渡せる |
| name | 名前(Const) |
| verify_shape | 値の形状のVarification(検証)をするか |

## Sample1

```python
import tensorflow as tf

const_a = tf.constant([1, 2, 3, 4, 5, 6], shape=(3,2))
print const_a

sess = tf.Session()
result_a = sess.run(const_a)
print result_a

```

結果
```shell
Tensor("Const_2:0", shape=(2, 3), dtype=float32)
[[1 2 3],
[4 5 6]]
```

## Sample2

```python
import tensorflow as tf

const_a = tf.constant(-1.0, shape=[2, 3])
print const_a

sess = tf.Session()
result_a = sess.run(const_a)
print result_a

```

結果
```shell
Tensor("Const_3:0", shape=(2, 3), dtype=float32)
[[-1. -1. -1.]
 [-1. -1. -1.]]
```



# 参考

[Constant](https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#constant)

