#!/bin/bash
export cwd_path='model_doctor-main/'
export middle_path=${cwd_path}'output/'
export model_name=$1
export data_name=$2
export exp_name=${1}'_'${2}
export in_channels=3
export num_classes=10
# export model_path=${middle_path}${exp_name}'/models/model_ori.pth'
# export data_path=${3}'/test'
export result_path=${middle_path}${exp_name}'/route_visual'
export result_name='route.jpg'
export grad_path=${middle_path}${exp_name}'/grads_50'
python ${cwd_path}core/model_decision_route_visualizing.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --grad_path ${grad_path} \
  --result_name ${result_name} \
  --result_path ${result_path}

# --model_path ${model_path} \
# --data_path ${data_path} \
