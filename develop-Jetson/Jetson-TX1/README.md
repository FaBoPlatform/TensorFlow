# 開発環境 Jetson TX1 JetPack 3.0
##### Ubuntu 16.04 LTS - TensorFlow r1.0.1
Ubuntu確認
```
cat /etc/lsb-release

DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=16.04
DISTRIB_CODENAME=xenial
DISTRIB_DESCRIPTION="Ubuntu 16.04.2 LTS"
```
カーネル確認
```
uname -a

Linux tegra-ubuntu 3.10.96-tegra #1 SMP PREEMPT Wed Nov 9 19:42:57 PST 2016 aarch64 aarch64 aarch64 GNU/Linux
```
アーキテクチャ確認
```
uname -m

aarch64
```

pip install用
[TensorFlow r1.0.1(CPU)](./r1.0.1/tensorflow-1.0.1-cp27-cp27mu-linux_aarch64.whl)

# TensorFlow build on Jetson TX1 JetPack 3.0
```
########################################
# JetPack L4T 3.0 (Ubuntu 16.04.02に更新)
########################################
いきなり更新してよい

PC VM UbuntuにJetPack3.0を入れて、あとは指示通りに進めばよい
VMで展開して、USB経由でJetsonに書き込む流れ

以下、JetPack L4T 3.0 更新済み


########################################
# 起動時毎回実行が必要になるもの
########################################
vi /etc/rc.local
####################
touch /tmp/jetpack.log
echo "starting script to send ip to host" >> /tmp/jetpack.log
/home/ubuntu/report_ip_to_host.sh &
echo "started script to send ip to host" >> /tmp/jetpack.log
# Launch CPU Fan
sh -c 'echo 255 > /sys/kernel/debug/tegra_fan/target_pwm'
# Overwrite DNS Server
cat <<EOF> /etc/resolv.conf
nameserver 8.8.8.8
EOF

exit 0
####################

login
user:ubuntu
password:ubuntu

Ubuntuターミナル出し方
Ctrl + Alt + T

日本語キーボードレイアウトに変更する
#setxkbmap -layout jp # これは一時的にしか効果が無い
#sudo dpkg-reconfigure keyboard-configuration # これは一時的にしか効果が無い

locale言語を英語に設定する
sudo dpkg-reconfigure locales


########################################
# WiFI IP固定設定
########################################
wpa_passphrase 'MySSID' 'MyPASSWORD' >> /etc/network/interfaces
表示されたssidとpskをwpa-ssidとwpa-pskとして記録する
(最初にvi /etc/network/interfacesをして、その後wpa_passphraseの結果をリダイレクト追記するといい)

vi /etc/network/interfaces
auto wlan0
iface wlan0 inet static
address 192.168.0.200
netmask 255.255.255.0
broadcast 192.168.0.255
gateway 192.168.0.1
dns-nameservers 8.8.8.8
wpa-ssid "MySSID"
wpa-psk d645bb286e786f7d16a7fb8ca750feaa7741f4c414a717f3f3476c3a30cbaf96

/etc/init.d/networking restart

########################################
# Ubuntu universalリポジトリを追加
########################################
sudo add-apt-repository universe
sudo apt-get update


########################################
# CPUファンを有効にする
########################################
# ref: http://elinux.org/Jetson/TX1_Controlling_Performance
sudo sh -c 'echo 255 > /sys/kernel/debug/tegra_fan/target_pwm'


########################################
# sshログインキーの作成
########################################
sudo su
mkdir /home/ubuntu/.ssh
ssh-keygen -t rsa
# /home/ubuntu/.ssh/Jetson_TK1_ubuntu_key
cd /home/ubuntu/.ssh
mv Jetson_TK1_ubuntu_key.pub authorized_keys
mv Jetson_TK1_ubuntu_key Jetson_TK1_ubuntu_key.pem
chmod 600 authorized_keys
chown -R ubuntu:ubuntu /home/ubuntu/.ssh


########################################
# Ubuntu 16.04 パッケージ更新
########################################
apt-get update
apt-get upgrade
apt-get dist-update
古くなったパッケージを削除する
sudo apt autoremove
reboot


########################################
# Python環境構築
########################################
# pipをインストールする
apt-cache search setuptools
Ubuntu universalリポジトリを追加をやっておくこと
apt-get install python-setuptools python3-setuptools python-pip python3-pip

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

apt-get install python-opencv python python-pkg-resources python-chardet python-colorama python-distlib python-six python-html5lib python-urllib3 python-requests python-setuptools python-pip python-wheel python-numpy python-pandas libcurl4-openssl-dev libprotobuf-dev software-properties-common swig libfreetype6-dev libpng-dev python-pyside python-pyqt5 python-qt4 curl

pip install --upgrade pip
pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade matplotlib
pip install --upgrade seaborn
pip install --upgrade requests
pip install --upgrade futures
# pipフル更新
pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U


########################################
# Java8 インストール
########################################
add-apt-repository "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main"
add-apt-repository ppa:webupd8team/y-ppa-manager
apt-get update
apt-get install oracle-java8-installer


########################################
# パッケージ作成準備
########################################
apt-get install devscripts


########################################
# bazel インストール
########################################
# https://bazel.build/versions/master/docs/install-compile-source.html
# bazel-0.4.5はaarch64をarmビルドに設定する必要がある
# https://github.com/dtrebbien/bazel/commit/eb01b38775ae9bfb554382fff429ecd3edfaf7d3
apt-get install openjdk-8-jdk pkg-config zip g++ zlib1g-dev unzip
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

mkdir ~/compile
cd ~/compile
uname -m
# sourceからのビルドはgitではなくzipで取得
# https://github.com/bazelbuild/bazel/releases
wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-dist.zip

unzip bazel-0.4.5-dist.zip
ls -l src/main/java/com/google/devtools/build/lib/util/CPU.java
chmod 644 src/main/java/com/google/devtools/build/lib/util/CPU.java
vi src/main/java/com/google/devtools/build/lib/util/CPU.java
####################
-  ARM("arm", ImmutableSet.of("arm", "armv7l")),
+  ARM("arm", ImmutableSet.of("aarch64", "arm", "armv7l")),
####################

./compile.sh
cp output/bazel /usr/local/bin/




########################################
# TensorFlow r1.0 GPUビルド
########################################
# USBメモリスティックをさしてswap領域を作る
sudo mkswap /dev/sda
sudo swapon /dev/sda
# swap止める時
sudo swapoff /dev/sda

git clone -b r1.0 --recurse-submodules https://github.com/tensorflow/tensorflow

cd tensorflow

uname -m
grep -Rl "arm"
grep -Rl "aarch64"

vi tensorflow/third_party/gpus/crosstool/CROSSTOOL.tpl
default_toolchain {
  cpu: "aarch64"
  toolchain_identifier: "local_linux"
}


cd ~/compile/tensorflow
./configure
Please specify the location of python. [Default is /usr/bin/python]: 
Please specify optimization flags to use during compilation [Default is -march=native]: aarch64
Do you wish to build TensorFlow with CUDA support? [y/N] 


####################
# CPU版ビルド
####################
# 時間がかかるので夜に実行放置
time bazel build -c opt --local_resources 2048,0.5,1.0 --verbose_failures //tensorflow/tools/pip_package:build_pip_package

whlを生成
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

TensorFlow インストール
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.0.1-cp27-cp27mu-linux_aarch64.whl 
```