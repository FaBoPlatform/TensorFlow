########################################
# OpenMPI v3.0.0 パッケージインストール
########################################
apt-get remove -y openmpi-common libopenmpi1.10
dpkg -i ../binary/openmpi-3.0.0.deb
