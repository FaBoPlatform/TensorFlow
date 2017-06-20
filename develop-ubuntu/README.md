# 開発環境 Ubuntu

##### Ubuntu 16.04 LTS - TensorFlow r1.2 GPU/XLA/MKL Python/C++
TensorFlow r1.2からIntel Math Kernel Libraryがサポートされるようになりました。AWS p2インスタンスはMKLに対応したCPUなので、MKLを有効にしたTensorFlowをビルドしたいと思います。
* [AWS EC2 p2.8xlarge Docker + git source compile (GPU)](./r1.2+c++/aws-ec2-docker-git-c++.md)

##### Ubuntu 16.04 LTS - TensorFlow r1.1.0 GPU/XLA Python/C++
TensorFlow r1.1.0はpip install --upgrade tensorflowで普通に使えるのですが、C++でも実行したいのでソースからビルドすることにします。
* [AWS EC2 p2.xlarge Docker + git source compile (GPU)](./r1.1.0+c++/aws-ec2-docker-git-c++.md)

##### Ubuntu 16.04 LTS - TensorFlow r1.0.1
TensorFlow r1.0になってからpip install tensorflowでインストール出来るようになったので、Dockerはjupyterを使いたい人や複数の環境を切り替えたい方向け。(GPU版はtensorflow-gpu)

* [Docker command](./docker-command.md)
* [AWS EC2 c4.large Docker (CPU)](./r1.0.1/aws-ec2-docker-cpu.md)
* [AWS EC2 p2.xlarge Docker (GPU)](./r1.0.1/aws-ec2-docker-gpu.md)
* [VM Docker (CPU)](./r1.0.1/vm-docker-cpu.md)
* [VM Docker Google Cloud Datalab (CPU)](./r1.0.1/vm-docker-datalab-cpu.md)
* [VM git source compile (CPU)](../android/build.md) - r1.0.0-rc2

