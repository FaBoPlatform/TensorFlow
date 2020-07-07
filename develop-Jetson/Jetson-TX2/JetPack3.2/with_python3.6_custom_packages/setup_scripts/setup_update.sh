########################################
# Ubuntu 16.04 パッケージ更新
########################################
# 認証なしCUDAパッケージになっていて、-yで更新出来ない
#302 upgraded, 9 newly installed, 0 to remove and 0 not upgraded.
#Need to get 90.3 MB/263 MB of archives.
#After this operation, 107 MB of additional disk space will be used.
#WARNING: The following packages cannot be authenticated!
#  libcudnn7-dev libcudnn7
#E: There were unauthenticated packages and -y was used without --allow-unauthenticated

apt-get update
time apt-get dist-upgrade -y --allow-unauthenticated
apt-get install -y htop
apt autoremove -y
