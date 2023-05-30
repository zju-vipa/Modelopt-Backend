#!/bin/bash
export cwd_path='model_doctor-main/'
export result_path=${cwd_path}'output/'
export model_name=$1
export data_name=$2
export exp_name=${1}'_'${2}
export num_layer=${4}
export in_channels=3
export num_classes=10
export model_path=${result_path}${exp_name}'/models/model_optim_cha_'${4}'.pth'
export data_path=${result_path}${exp_name}'/images_50_optim_cha_'${4}''
export grad_path=${result_path}${exp_name}'/grads_50_optim_cha_'${4}''
export theta=0.2
export device_index='2'
python ${cwd_path}core/grad_calculate.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --grad_path ${grad_path} \
  --theta ${theta} \
  --device_index ${device_index}
