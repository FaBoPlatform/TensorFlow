# Local環境の整備

TensorFlowの実行は、CloudMLでできるが、matplotlibは、CloudML上では、画像に出力はできてもCloudShell上にWindowを出して表示できる事ができないので、matplotlibとnumpyが整備されたPython環境をLocalに構築しておく。

Python 3.x系のANACONDAをインストールします。

[https://www.continuum.io/downloads](https://www.continuum.io/downloads)

```shell
$ python --version
```

で、うまくインストールできたかを確認します。