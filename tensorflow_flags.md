# コマンドライン引数の処理

`tf.app.flags`はコマンドライン引数を処理するための**Google commandline flags**のラッパーモジュールで、データセットのパス指定やデバッグ等のユーティリティに用いられる。

## Sample

```python
# coding:utf-8
"""
file name : tensorflow_flags.py
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# 文字列の定義
tf.app.flags.DEFINE_string("name", None, "名前")
# 整数の定義
tf.app.flags.DEFINE_integer("int_value", 5, "整数値")
# 浮動小数点数の定義
tf.app.flags.DEFINE_integer("float_value", 3.14, "浮動小数点数")
# ブール値
tf.app.flags.DEFINE_bool("bool_value", False, "ブール値")

if __name__ == "__main__":
    print FLAGS.name
    print FLAGS.int_value
    print FLAGS.float_value
    print FLAGS.bool_value
```

実行結果 :

`python tensorflow_flags.py` :

```
None
5
3.14
False
```

`python tensorflow_flags.py -h` :

```
usage: tensorflow_flags.py [-h] [--name NAME] [--int_value INT_VALUE]
                           [--float_value FLOAT_VALUE]
                           [--bool_value [BOOL_VALUE]] [--nobool_value]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           名前
  --int_value INT_VALUE
                        整数値
  --float_value FLOAT_VALUE
                        浮動小数点数
  --bool_value [BOOL_VALUE]
                        ブール値
  --nobool_value
```

`python tensorflow_flags.py --name feynman --int_value 7 --bool_value` :

```
feynman
7
3.14
True
```

## 参考

* [python-gflags](https://github.com/google/python-gflags)
