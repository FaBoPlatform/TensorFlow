JetPack 3.2 TensorFlow環境構築スクリプト

JetPack 3.2インストール直後を起点とする。
Python 3.6
OpenCV 3.4.0
TensorFlow r1.6.0


1. sudo su
2. chmod 755 ../setup_scripts/*.sh
3. chmod 755 ../install_scripts/*.sh
4. ../setup_scripts/setup.sh
# wait reboot

5. ./install.sh
# package: wait 2 hours (default)
# build: wait 6-7 hours (edit install.sh)
