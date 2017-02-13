# Tensorflowのグラフ操作 Part2

## グラフを取得する

**構築されたグラフを取得する方法：**

* `tf.get_default_graph()`
* `sess.graph`

```python
# coding:utf-8
import tensorflow as tf
# 足し算を行うグラフを構築
a = tf.constant(1, name="a")
b = tf.constant(2, name="b")
add_op = tf.add(a, b, name="add_op")
# 方法(1)
g1 = tf.get_default_graph()
print(g1)
# 方法(2)
sess = tf.Session()
g2 = sess.graph
print(g2)
```

### 実行結果

同一の`graph`オブジェクトが取得されていることが分かる。

```
<tensorflow.python.framework.ops.Graph object at 0x102541350>
<tensorflow.python.framework.ops.Graph object at 0x102541350>
```

---

## 複数のグラフを同じプログラムで扱う

足し算を行うグラフと引き算を行うグラフの2つを同じプログラムで扱う。

**複数のグラフを扱うための手順：**

1. `tf.Graph()`で`graph`オブジェクトの生成
2. `tasizan_graph.as_default()`で対象となるグラフを指定
3. グラフの構築
4. `tf.Session(graph=...)`で実行するグラフを指定

```python
# coding:utf-8
import tensorflow as tf
# 足し算用グラフ
tasizan_graph = tf.Graph()
# 引き算用グラフ
hikizan_graph = tf.Graph()
# 引き算グラフの構築
with tasizan_graph.as_default():
    a = tf.placeholder(tf.int32, shape=[], name="a")
    b = tf.placeholder(tf.int32, shape=[], name="b")
    add_op = tf.add(a, b, name="add_op")
# 足し算グラフの構築
with hikizan_graph.as_default():
    x = tf.placeholder(tf.int32, shape=[], name="x")
    y = tf.placeholder(tf.int32, shape=[], name="y")
    sub_op = tf.sub(x, y, name="sub_op")
# 足し算の実行
with tf.Session(graph=tasizan_graph) as sess:
    ret = sess.run(add_op, feed_dict={a:1,b:1})
    print ret
# 引き算の実行
with tf.Session(graph=hikizan_graph) as sess:
    ret = sess.run(sub_op, feed_dict={x:1,y:1})
    print ret
```

### 実行結果

```
2
0
```

---

# 参考