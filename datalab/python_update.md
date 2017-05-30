# Python2からPython3.x系へのアップデート
pyenvを使用してアップデートする

起動しているDockerのコンテナのシェルにログインする  
NAMEには起動中のコンテナIDかNameを指定すること   
```shell
$ docker exec -it NAME /bin/bash
```
### パッケージのインストール
最初にapt-getを更新する
```shell
apt-get upgrade
```
パッケージをインストール
```shell
apt-get install git gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev
```
pyenvをインストールする
```shell
git clone git://github.com/yyuu/pyenv.git ~/.pyenv
```
環境変数を設定する  
`echo $SHELL`　で使用しているシェルを確認し、そのシェルに以下を追加する  
下の例だと`bash`を使用していたので`vi ~/.bashrc`か`vim ~/.bashrc`にて追加した
```shell
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
```
追加したら読み込み、whichでパスが出ることを確認する  
```shell
source ~/.bashrc
which pyenv
```

## Pythonの別のバージョンのインストール  
`pyenv install -l`にてインストール可能なバージョンを確認する  
欲しいバージョンがあったらそれをインストールする
ここでは例として3.6.1をインストールしてみる   
```shell
pyenv install 3.6.1
# インストールされたかを確認する
pyenv versions
```
インストールが無事完了したら
`pyenv global 3.6.1`で使用するpythonのバージョンを切り替える  
他のバージョンをインストールしたらそのバージョンを指定すること  
`python -V`で指定したバージョンが出力されることを確認する  

最後に、アップデートした状態をDockerを保存する。
ローカルで`docker ps`で、起動しているDockerコンテナ一覧を取得する。

```shell
$ docker ps
```

コンテナIDをコピーし、下記のコマンドで変更をコミットする

```shell
$ docker commit -m "update the python2 to python3" 19627749df78 tensorflow_python3
```

|引数|意味|
|:--|:--|
|-m "update the python2 to python3" | 好きなコメントを記載|
|19627749df78| `docker ps`で確認した各自のCONTAINER ID|
|tensorflow_python3|任意のdockerイメージの名前|
