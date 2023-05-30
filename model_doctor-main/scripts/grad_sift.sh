#!/bin/bash
export cwd_path='model_doctor-main/'
export middle_path=${cwd_path}'output/'
export model_name=$1
export data_name=$2
export exp_name=${1}'_'${2}
export in_channels=3
export num_classes=10
export model_path=${middle_path}${exp_name}'/models/model_ori.pth'
# export data_path=${cwd_path}'datasets/'$2'/test'
export data_path=${3}'/test'
export result_path=${middle_path}${exp_name}'/sift_visual'
python ${cwd_path}core/grad_sift.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --result_path ${result_path}
