########################################
# Java8 レポジトリ追加
########################################
add-apt-repository -y ppa:webupd8team/java
apt-get update

########################################
# Java8 インストール
########################################
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections && apt-get install -y oracle-java8-installer
