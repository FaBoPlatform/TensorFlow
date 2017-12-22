########################################
# Build Tools Install
########################################
apt-get update
apt-get install -y zip git curl locate libeigen3-dev libprotobuf-dev
updatedb
locate libcuda.so

# /usr/bin/find: '/run/user/106/gvfs': Permission denied
# と表示されるが、これはGnome用のGVFSのFUSEインターフェースなので無視する。
