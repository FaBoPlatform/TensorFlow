# Boolean tensor

`tf.where(input, name=None)`

Trueとなっているインデックスを返す

サンプルコード :

```python
# coding:utf-8
from random import choice
import numpy as np
import tensorflow as tf

# TrueかFalseを持つ4x4行列
input_data = np.array([choice([True,False]) for _ in range(16)]).reshape(4, 4)
where_op = tf.where(input_data)

with tf.Session() as sess:
    print input_data
    # TrueとなっているTensorのインデックスを返す
    w = sess.run(where_op)
    print w
```

実行結果 :

```
[[ True  True False False]
 [ True False  True False]
 [ True  True  True False]
 [False  True False  True]]
[[0 0]
 [0 1]
 [1 0]
 [1 2]
 [2 0]
 [2 1]
 [2 2]
 [3 1]
 [3 3]]
```
