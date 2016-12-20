# キャスト

Tensorの値を変換する

|関数名|説明|
|:-:|:-:|
|`tf.cast(x, dtype)`|キャストする|
|`tf.string_to_number(string_tensor)`|文字列を`tf.float32`にキャストする|
|`tf.to_float(x)`|`tf.float32`にキャストする|
|`tf.to_int32(x)`|`tf.int32`にキャストする|

サンプルコード :

```python
# coding:utf-8
import numpy as np
import tensorflow as tf

int_tensor = tf.constant([-1, 0, 2, 10], dtype=tf.int32)
int_tensor_cast = tf.cast(int_tensor, dtype=tf.float32)
# 以下も同様
# int_tensor_cast = tf.to_float(int_tensor)

# 文字列Tensor
string_tensor = tf.constant(["0.123", "3.14"], dtype=tf.string)
# 文字列Tensorを浮動小数点数にキャストする
string_tensor_cast = tf.string_to_number(string_tensor)

# Bool値Tensor
bool_tensor = tf.constant([True, False], dtype=tf.bool)
# Bool値Tensorを整数値にキャストする
bool_tensor_cast = tf.to_int32(bool_tensor)

with tf.Session() as sess:
    print int_tensor.eval()
    print int_tensor.dtype
    int_tensor = int_tensor_cast.eval()
    print int_tensor
    print int_tensor.dtype

    print string_tensor.eval()
    print string_tensor_cast.eval()

    print bool_tensor.eval()
    print bool_tensor_cast.eval()
```

実行結果 :

```
[-1  0  2 10]
<dtype: 'int32'>
[ -1.   0.   2.  10.]
float32
['0.123' '3.14']
[ 0.123      3.1400001]
[ True False]
[1 0]
```

## 参考

* [TensorFlow API](https://www.tensorflow.org/api_docs/python/array_ops/casting)
