########################################
# bazel-0.5.4 ビルド # 最新は0.7.0だけどTensorflow公式テストは0.5.4
########################################
mkdir -p /compile \
&& cd /compile \
&& wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-dist.zip \
&& unzip bazel-0.5.4-dist.zip -d bazel-0.5.4 \
&& cd bazel-0.5.4 \
&& ./compile.sh \
&& cp -f output/bazel /usr/local/bin/

