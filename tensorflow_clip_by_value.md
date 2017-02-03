# Tensorのクリッピング

Tensorを指定した範囲に収まるようにする

`tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)`

* `t` : A Tensor.
* `clip_value_min`: 範囲の最小値となるスカラー値
* `clip_value_max`: 範囲の最大値となるスカラー値

## Sample

![](/img/clip01.png)

![](/img/clip02.png)

## 参考

* https://www.tensorflow.org/api_docs/python/train/gradient_clipping#clip_by_value

## Notebook

[https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/clip.ipynb](https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/clip.ipynb)
