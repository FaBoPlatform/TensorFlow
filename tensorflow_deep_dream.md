# DeepDream

GoogLeNetの各レイヤーを可視化する

コマンドライン引数 :

* `--layer_list` : GoogLeNetのレイヤー一覧
* `--random_noise` : ランダムノイズ画像に対してDeepDreamを適用する
* `--layer` : 可視化するレイヤー
* `--channel` : 可視化するレイヤーのチャネル
* `--in_img` : 入力画像
* `--out_img` : 画像の出力先(拡張子は省略する)

モデルおよびセッションデータのダウンロード :

```
$ wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
$ unzip inception5h.zip
```

サンプルコード :

```python
# coding:utf-8
"""
filename: tensorflow_deep_dream.py
"""
import numpy as np
import PIL.Image
import tensorflow as tf

# 結果が同じになるように、乱数のシードを設定する
tf.set_random_seed(20200724)
np.random.seed(20200724)

# コマンドライン引数の定義
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool("layer_list", False, "GoogLeNetのレイヤー一覧を表示する")
tf.app.flags.DEFINE_bool("random_noise", False, "ランダムノイズ画像を使用する")
tf.app.flags.DEFINE_string("layer", None, "画像生成に使うレイヤー名")
tf.app.flags.DEFINE_integer("channel", None, "画像生成に使うレイヤーのチャネル")
tf.app.flags.DEFINE_string("in_img", None, "入力画像のパス")
tf.app.flags.DEFINE_string("out_img", "deep_dream.jpg", "出力画像のパス 画像の拡張子は要省略")

# モデルおよびセッション情報を格納したファイル
model_fn = 'tensorflow_inception_graph.pb'
# モデルの構築とセッションの読み込み
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
# pbファイルを読み込む
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 画像を格納するTensor
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# 各層の名前を出力する
def show_layer_list():
    """レイヤー一覧を表示する"""
    for op in graph.get_operations():
        if op.type=='Conv2D' and 'import/' in op.name:
            print op.name

def read_image(name):
    """画像を読み込み、nparrayに変換する"""
    img_arr = PIL.Image.open(name)
    return np.array(img_arr, dtype=np.float32)

def save_image(a, name):
    """nparrayを画像として保存する"""
    x = np.uint8(np.clip(a, 0.0, 1.0) * 255.0)
    img = PIL.Image.fromarray(x)
    img.save(name)

def visstd(a, s=0.1):
    """画像の正規化する"""
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

def T(layer):
    """指定した層の出力Tensorを取得する"""
    return graph.get_tensor_by_name("import/%s:0" % layer)

def render_naive(t_obj, iter_n=20, step=1.0, name="noise_dream.jpg"):
    """ランダムノイズ画像に対してDeepdreamを適用する"""
    img0 = np.random.uniform(size=(224,224,3)) + 100.0
    # 最適化を行うTensor
    t_score = tf.reduce_mean(t_obj)
    # 自動微分の累乗
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # 勾配を正規化する
        g /= g.std() + 1e-8
        img += g * step
    save_image(visstd(img), name)

def tffunc(*argtypes):
    """TFグラフ関数のヘルパー関数"""
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def resize(img, size):
    """TensorFlowによって画像をリサイズする"""
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
# tf関数化
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    """入力画像に対する勾配を計算する"""
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    # ランダムに配列の要素を回転させる
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2,sz), sz):
        for x in range(0, max(w-sz//2,sz), sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, name="deep_dream"):
    """ノイズ画像に対して Deep dream画像を生成する"""
    # 最適化を行うTensor
    t_score = tf.reduce_mean(t_obj)
    # Tensorの勾配
    t_grad = tf.gradients(t_score, t_input)[0]

    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    # オクターブごとに画像を生成する
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
        save_image(img/255.0, "%s_%i.jpg"%(name,octave))

if __name__ == "__main__":
    output_path = FLAGS.out_img
    if FLAGS.layer_list:
        show_layer_list()
        quit()
    if FLAGS.random_noise:
        layer = "mixed4d_3x3_bottleneck_pre_relu"
        tensor = T(layer)[:,:,:,139]
        render_naive(tensor, name=output_path)
        quit()
    if FLAGS.in_img:
        # 入力画像
        img = read_image(FLAGS.in_img)
        tensor = tf.square(T('mixed4c'))
        if FLAGS.layer and FLAGS.channel:
            tensor = T(FLAGS.layer)[:,:,:,FLAGS.channel]
        render_deepdream(tensor, img, name=output_path)
```

`python tensorflow_deep_dream.py --layer_list`の実行結果 :

```
import/conv2d0_pre_relu/conv
import/conv2d1_pre_relu/conv
import/conv2d2_pre_relu/conv
import/mixed3a_1x1_pre_relu/conv
import/mixed3a_3x3_bottleneck_pre_relu/conv
import/mixed3a_3x3_pre_relu/conv
import/mixed3a_5x5_bottleneck_pre_relu/conv
import/mixed3a_5x5_pre_relu/conv
...略...
import/mixed5b_1x1_pre_relu/conv
import/mixed5b_3x3_bottleneck_pre_relu/conv
import/mixed5b_3x3_pre_relu/conv
import/mixed5b_5x5_bottleneck_pre_relu/conv
import/mixed5b_5x5_pre_relu/conv
import/mixed5b_pool_reduce_pre_relu/conv
import/head0_bottleneck_pre_relu/conv
import/head1_bottleneck_pre_relu/conv
```

`python tensorflow_deep_dream.py --random_noise`の実行結果 :

![](/img/deep_dream_01.jpg)

元画像のダウンロード :

```
$ wget https://github.com/FaBoPlatform/TensorFlow/raw/master/img/scenery.jpgg
```

`python tensorflow_deep_dream.py --in_img scenery.jpg --out_img scenery_deep`の実行結果 :

元画像

![](/img/scenery.jpg)

Deepdreamを適用した画像

![](/img/deep_dream_02.jpg)

`python tensorflow_deep_dream.py --in_img scenery.jpg --out_img scenery_deep --layer mixed3b_1x1_pre_relu --channel 101`の実行結果 :

![](/img/deep_dream_03.jpg)

## 参考

* [DeepDreaming with TensorFlow](https://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
