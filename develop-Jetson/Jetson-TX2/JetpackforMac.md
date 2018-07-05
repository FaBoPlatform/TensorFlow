## MacでのJetson Tx2 に対するJetPack3.X のインストール方法      

Ubuntuで作業を行う必要があるため、`Virtual Box`,`VMware`などをインストールする必要がある。`Virtual Box`のパッケージは下記のサイトでダウンロードできる。        

[https://www.virtualbox.org/wiki/Downloads](https://www.virtualbox.org/wiki/Downloads)          

`Virtual Box`のインストールが終了したら下記のサイトでUbuntuのisoイメージをダウンロードする。(ubuntu-ja-16.04-desktop-amd64.isoで動くことを確認)         
[http://www.ubuntulinux.jp/download/ja-remix](http://www.ubuntulinux.jp/download/ja-remix)                

isoイメージのダウンロードが終了したら`Virtual Box`を起動し、左上の`新規`ボタンをクリックし以下のような設定を行う         
![](/img/jetson4mac1.png)       
名前とメモリーサイズは任意の設定          

作成が終了したら作成したものを起動し、その際に事前にダウンロードしたisoイメージを設定して`Start`ボタンを押す。            
![](/img/jetson4mac2.png)         

Ubuntuが起動するので、セットアップを行う。                 

作成した仮想マシンの`設定`を開き、`ポート > USB`に移動する。          
`USBコントローラーを有効化`において、`USB2.0`のチェックを`USB3.0`に必ず変更する。    

![](/img/jetson4mac3.png)         

後は下記のサイトに従い、JetPackのインストールを行う。         
[http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.1/jetpack_l4t_install.htm](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.1/jetpack_l4t_install.htm)
