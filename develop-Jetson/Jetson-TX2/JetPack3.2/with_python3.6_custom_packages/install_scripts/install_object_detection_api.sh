########################################
# TensorFlow Object Detection API
########################################

mkdir -p /home/ubuntu/notebooks/github
cd /home/ubuntu/notebooks/github
git clone https://github.com/tensorflow/models

cd models
git checkout -b r1.5

cd research

apt-get install -y protobuf-compiler python3-tk
protoc object_detection/protos/*.proto --python_out=.
pip3 install .

#cd /home/ubuntu/notebooks/github/models/research

env PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim python object_detection/builders/model_builder_test.py


# From models directory
python object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`


env PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v1_pets.config \
    --checkpoint_dir=object_detection/ssd_mobilenet_v1_coco_2017_11_17 \
    --eval_dir=object_detection/data
    --label_map_path=object_detection/data/pet_label_map.pbtxt

/home/ubuntu/notebooks/github/models/research/object_detection/data/pet_label_map.pbtxt


python object_detection/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
