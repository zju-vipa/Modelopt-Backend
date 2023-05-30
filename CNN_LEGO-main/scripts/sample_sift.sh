#!/bin/bash
export result_path='output/'
export exp_name='vgg16_cifar10'
export model_name='vgg16'
export data_name='cifar10'
export num_classes=10
export model_path=${result_path}${exp_name}'/models/model_ori.pth'
export data_path='/nfs4-p1/gj/datasets/cifar10/train'
export image_path=${result_path}${exp_name}'/images'
export num_images=50
export device_index='0'
python core/sample_sift.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --image_path ${image_path} \
  --num_images ${num_images} \
  --device_index ${device_index}
