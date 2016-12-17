# Tensorのクリッピング

Tensorを指定した範囲に収まるようにする

`tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)`

* `t` : A Tensor.
* `clip_value_min`: 範囲の最小値となるスカラー値
* `clip_value_max`: 範囲の最大値となるスカラー値

サンプルコード :

```python
# coding:utf-8
import tensorflow as tf

with tf.Session() as sess:
    p = tf.constant([[0.0,0.25],[0.5,1.0]], dtype=tf.float32, shape=(2,2))
    # 値を指定した範囲の中に収まるようにする
    value = tf.clip_by_value(p, 1e-10, 1.0)
    print p.eval()
    print value.eval()
```

実行結果 :

```
[[ 0.    0.25]
 [ 0.5   1.  ]]
[[  1.00000001e-10   2.50000000e-01]
 [  5.00000000e-01   1.00000000e+00]]
```

## 参考

* [TensorFlow API](https://www.tensorflow.org/api_docs/python/train/gradient_clipping#clip_by_value)
