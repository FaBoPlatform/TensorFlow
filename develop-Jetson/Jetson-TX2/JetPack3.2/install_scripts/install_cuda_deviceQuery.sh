########################################
# CUDA deviceQueryビルド
########################################
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
mkdir -p /usr/local/cuda/extras/demo_suite
ln -s /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery /usr/local/cuda/extras/demo_suite

/usr/local/cuda/extras/demo_suite/deviceQuery
