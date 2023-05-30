#!/bin/bash
export cwd_path='model_doctor-main/'
export result_path=${cwd_path}'output/'
export model_name=$1
export data_name=$2
export exp_name=${1}'_'${2}
export in_channels=3
export num_classes=10
export num_epochs=20
export model_dir=${result_path}${exp_name}'/models'
export data_dir=${3}
export log_dir=${result_path}'/runs/'${exp_name}
export device_index='2'
python ${cwd_path}train.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --num_epochs ${num_epochs} \
  --model_dir ${model_dir} \
  --data_dir ${data_dir} \
  --log_dir ${log_dir} \
  --device_index ${device_index}
