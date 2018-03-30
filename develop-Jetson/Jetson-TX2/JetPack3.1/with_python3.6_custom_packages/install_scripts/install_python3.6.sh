########################################
# Python3.6 レポジトリ追加
########################################
add-apt-repository -y ppa:jonathonf/python-3.6
apt-get update

########################################
# Python3.6 インストール
########################################
apt-get install -y python3.6 python3.6-dev
update-alternatives --install /usr/bin/python3 python /usr/bin/python3.6 0
apt-get install -y python3-pip
rm -rf /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python

########################################
# add-apt-repository コマンドのpythonバージョンを3.5に指定する
########################################
# sed
# escape characters \'$.*/[]^
# 1. Write the regex between single quotes.
# 2. \ -> \\
# 3. ' -> '\''
# 4. Put a backslash before $.*/[]^ and only those characters.

# before
#! /usr/bin/python3
# after
#! /usr/bin/python3.5

sed -i 's/#! \/usr\/bin\/python3$/#! \/usr\/bin\/python3\.5/g' /usr/bin/add-apt-repository
