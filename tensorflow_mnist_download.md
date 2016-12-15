# MNISTデータセットのダウンロード

MNIST(手書き数字)データセットをダウンロードする

```python
from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータセットの読み込み
# 指定ディレクトにデータがない場合はダウンロード
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
```

実行結果

```
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting ./MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting ./MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting ./MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting ./MNIST_data/t10k-labels-idx1-ubyte.gz
```

ローカルにデータセットが保存されている

```
$ ls ./MNIST_data
t10k-images-idx3-ubyte.gz  train-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz  train-labels-idx1-ubyte.gz
```

## 参考

* [MNIST For ML Beginners](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/)
