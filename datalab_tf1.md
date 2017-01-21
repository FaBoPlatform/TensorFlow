
# VIMをインストール

```shell
$ apt-get update
$ apt-get install vim
```

# Javaのインストール

```shell
$ vim /etc/apt/sources.list.d/java-8-debian.list
```

java-8-debian.list
```shell
deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main
deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main
```

```shell
$ apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886
```

```shell
$ apt-get update
$ apt-get install oracle-java8-installer
```

```shell
$ javac -version
$ javac 1.8.0_121
```

# Bazelのインストール

```shell
$ wget https://github.com/bazelbuild/bazel/releases/download/0.4.3/bazel-0.4.3-installer-linux-x86_64.sh
$ chmod +x bazel-0.4.3-installer-linux-x86_64.sh
$ ./bazel-0.4.3-installer-linux-x86_64.sh
```

# TensorFlow

```shell
$ apt-get install libcurl3-dev
$ git clone -b v1.0.0-alpha https://github.com/tensorflow/tensorflow
$ ./configure
```




