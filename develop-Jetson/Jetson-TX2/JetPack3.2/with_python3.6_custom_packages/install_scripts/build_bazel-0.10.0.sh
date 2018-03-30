########################################
# bazel-0.10.0 ビルド # 最新は0.7.0だけどTensorflow公式テストは0.10.0
########################################
mkdir -p /compile \
&& cd /compile \
&& wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip \
&& unzip bazel-0.10.0-dist.zip -d bazel-0.10.0 \
&& cd bazel-0.10.0 \
&& ./compile.sh \
&& cp -f output/bazel /usr/local/bin/

