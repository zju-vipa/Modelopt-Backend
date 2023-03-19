# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/8/26 10:06
# @Email：329201962@qq.com

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
import shutil
import sys
import numpy as np
from flask import Flask, render_template, redirect, request, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from common.getConfig import getConfig
from common.new_alchemy_encoder import new_alchemy_encoder
####

app = Flask(__name__,
            template_folder="../frontend/dist/",
            static_folder="../frontend/dist/static/")

# 获取mysql配置信息
HOSTNAME = getConfig("config", 'mysql', 'host')  # def getConfig(filename, section, option):
PORT = getConfig("config", 'mysql', 'port')
USERNAME = getConfig("config", 'mysql', 'user')
PASSWORD = getConfig("config", 'mysql', 'password')
DATABASE = getConfig("config", 'mysql', 'db')

# 文件存储url配置信息
MODEL_URL = getConfig("config", "save_url", "model")
DATA_URL = getConfig("config", "save_url", "data")
PICS_URL = getConfig("config", "save_url", "pics")
MARK_URL = getConfig("config", "save_url", "mark")
RESULT_URL = getConfig("config", "save_url", "result")
WEIGHT_URL = getConfig("config", "save_url", "weight")

# 配置指定的文件名
PICS_NAME = getConfig("config", "specify_file_name", "pics")
WEIGHT_NAME = getConfig("config", "specify_file_name", "weight")
# 以上从config文件读取
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(USERNAME, PASSWORD,
                                                 HOSTNAME, PORT, DATABASE)
# 数据库的url
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI  # configuration录入数据库
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # 数据库迁移
'''
    使用flask_migrate把数据模型映射到MySql数据中
    打开Pycharm的Terminal输入命令:flask db init->flask db migrate->flask db upgrade
'''
'''
[save_url]
model = ../storage/model/
data = ../storage/dataset/
pics = ../storage/pics/
mark = ../storage/mark/
weight = ../storage/weight/
result = ./static/result
'''


class Model(db.Model):
    __tablename__ = 'model'  # 表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    model_url = db.Column(db.String(255), nullable=False)  # 四列数据构造


class Data(db.Model):
    __tablename__ = 'data'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.String(255), nullable=False)
    data_name = db.Column(db.String(255), nullable=False)
    data_url = db.Column(db.String(255), nullable=False)


class Task(db.Model):
    __tablename__ = 'task'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.String(255), nullable=False)
    model_id = db.Column(db.Integer, nullable=False)
    data_id = db.Column(db.Integer, nullable=False)
    weight_url = db.Column(db.String(255), nullable=False)


'''
    模型页面
    1.进去刷新列表 GET /modeldoctor/model
    2.上传模型（检验模型是不是合法）POST /modeldoctor/model
'''


@app.route('/modeldoctor/model', methods=['get'])
def get_model_list():
    models = Model.query.all()  # 查询数据库所有项
    models = json.dumps(models, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    return models


@app.route('/modeldoctor/model', methods=['post'])
def add_model():
    print(request.files)
    file = request.files['file']
    file.save('{}{}'.format(MODEL_URL, file.filename))
    model = Model(time=datetime.now(), model_name=file.filename.split(".")[0], model_url='{}{}'.format(MODEL_URL, file.filename))
    db.session.add(model)
    db.session.commit()  # 上传数据库项
    return redirect('/')  # 重定向


'''
    数据集页面：
    1.进去刷新列表 GET  /modeldoctor/dataset
    2.上传数据集（检验数据集是不是合法）POST /modeldoctor/dataset
'''


@app.route('/modeldoctor/dataset', methods=['get'])
def get_data_list():
    data = Data.query.all()  # 查询Data表
    data = json.dumps(data, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    return data  # 返回json数据


@app.route('/modeldoctor/dataset', methods=['post'])
def add_data():
    file = request.files['file']
    print("0")
    print(request.files['file'])
    file.save('{}{}'.format(DATA_URL, file.filename))
    print("0")
    #zip_file = zipfile.ZipFile(file)
    # 解压
   # file2 = zip_file.extractall(DATA_URL)
    # file2.close()

    date = Data(time=datetime.now(), data_name=file.filename.split(".")[0], data_url='{}{}'.format(DATA_URL, file.filename))
    print("0")
    db.session.add(date)
    db.session.commit()  # 上传数据集并add到数据表
    return redirect('/')


'''
    任务页面：
    1.开始第一阶段训练*  PUT /modeldoctor/task/step1
    2.下载第一阶段的图片（始终一个文件夹）GET /modeldoctor/task/pics
    3.上传标注 POST /modeldoctor/task/mark
    4.开始第二阶段训练*  PUT /modeldoctor/task/step2
    5.输出诊断结果(返回图片的地址list) GET /modeldoctor/task/result
    6.下载模型优化权重(分文件夹存储) GET /modeldoctor/task/weigtht
'''


@app.route('/word/cloud/generate', methods=['post'])
def test():
    print("DDDD")
    return redirect('/')


'''
export result_path='../output/'
export model_name=$1
export data_name=$2
export exp_name=${model_name}+'_'+${data_name}+'_22051035'
export in_channels=3
export num_classes=10
export num_epochs=10
export model_dir=${result_path}${exp_name}'/models'
export data_dir='../datasets/'${data_name}'/images'
export log_dir=${result_path}'/runs/'${exp_name}
export device_index='0'
python ../train.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --num_epochs ${num_epochs} \
  --model_dir ${model_dir} \
  --data_dir ${data_dir} \
  --log_dir ${log_dir} \
  --device_index ${device_index}.
'''


@app.route('/modeldoctor/task/step1', methods=['post'])
def first_run():
    # 治疗的工具参数
    focus_on_the_sample = request.json.get('focus_on_the_sample')
    print("focus_on_the_sample:", focus_on_the_sample)
    focus_on_the_component = request.json.get('focus_on_the_component')
    print("focus_on_the_component:", focus_on_the_component)
    focus_on_the_path1 = request.json.get('focus_on_the_path1')
    print("focus_on_the_path1:", focus_on_the_path1)    # 缺陷诊断的工具参数
    focus_on_the_primitive_character = request.json.get('focus_on_the_primitive_character')
    focus_on_the_path2 = request.json.get('focus_on_the_path2')
    # 错误原因显示参数
    cause = request.json.get('cause')
    # 治疗部位
    treatment = request.json.get('treatment')

    # 模型和数据
    model_id = request.json.get('model_id')
    data_id = request.json.get('data_id')
    print("model_id:", model_id)
    print("data_id:", data_id)
    model = Model.query.filter_by(id=model_id).first().model_url
    data = Data.query.filter_by(id=data_id).first().data_url  # 模型参数
    model_name = Model.query.filter_by(id=model_id).first().model_name
    data_name = Data.query.filter_by(id=model_id).first().data_name
    result_path = '../model_doctor-main/output/'
   # model_name='alexnet'
   # data_name='cifar10'
    exp_name = model_name + '_' + data_name + '_22051035'
    in_channels = 3
    num_classes = 10
    num_epochs = 10
    # model_dir =${result_path}${exp_name}'/models'
    model_dir = os.path.join(result_path, exp_name, 'models')
    data_dir = os.path.join('../model_doctor-main/datasets', data_name, 'images')
    log_dir = os.path.join(result_path, 'runs', exp_name)
    device_index = '0'
    os.system("python ../model_doctor-main/train.py --model_name model_name --data_name data_name --in_channels in_channels --num_classes num_classes --num_epochs num_epochs --model_dir model_dir --data_dir data_dir --log_dir log_dir --device_index device_index")
    model_path = os.path.join(result_path, exp_name, 'models/model_ori.pth')
    data_path = os.path.join('../model_doctor-main/datasets', data_name, 'images/train')
    image_path = os.path.join(result_path, exp_name, 'images_50')
    num_images = 50
    os.system("python ../model_doctor-main/core/image_sift.py --model_name model_name --data_name data_name --in_channels in_channels --num_classes num_classes --model_path model_path --data_path data_path --image_path image_path --num_images num_images --device_index device_index")
    data_path = os.path.join(result_path, exp_name, 'images_50')
    grad_path = os.path.join(result_path, exp_name, 'grads_50')
    theta = 0.2
    os.system("python ../model_doctor-main/core/grad_calculate.py --model_name model_name --data_name data_name --in_channels in_channels --num_classes num_classes --model_path model_path --data_path data_path --grad_path grad_path --theta theta --device_index device_index")

    print("QQQQQQ")
    #result1=os.system("sh ../model_doctor-main/scripts/train.sh "+model_name+" "+data_name)
    # print("train.sh:",result1)
    #result2 = os.system("sh ../model_doctor-main/scripts/image_sift.sh " + model_name + " " + data_name)
   # print("image_sift.sh:", result2)
    #result3 = os.system("sh ../model_doctor-main/scripts/grad_calculate.sh " + model_name + " " + data_name)
   # print("grad_calculate.sh:", result3)
    R = os.path.join('../model_doctor-main/output', model_name+"_"+data_name+"_22051035", 'grads_50')

    if not os.path.exists(RESULT_URL):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(RESULT_URL)

    if os.path.exists(R):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(R):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, RESULT_URL)
               # print(src_file)
    print('copy dir finished!')
    ####  result4 = os.system("python ../model_doctor-main/preprocessing/labelme_to_mask.py")
    ####  print("labelme_to_mask.py:", result4)

    '''
    此处利用以上参数进行训练，待补充
    '''
    # 训练结束后，保存待标记的图片 和诊断图片 分别存到 PICS_URL、RESULT_URL
    '''
    待补充
    '''
    return redirect('/')


@app.route('/modeldoctor/task/pics', methods=['get'])
def get_pics():  # 返回图片，下载
    return send_from_directory('{}'.format(PICS_URL), '{}'.format(PICS_NAME), as_attachment=True)


@app.route('/modeldoctor/task/mark', methods=['post'])
def add_mark():  # 标注上传到MARK URL中（一个文件夹）
    file = request.files.get('mark')
    file.save('{}{}'.format(MARK_URL, file.filename))
    return redirect('/')


@app.route('/modeldoctor/task/step2', methods=['put'])
def second_run():
    # 治疗的工具参数
    focus_on_the_sample = request.json.get('focus_on_the_sample')
    focus_on_the_component = request.json.get('focus_on_the_component')
    focus_on_the_path1 = request.json.get('focus_on_the_path1')
    # 缺陷诊断的工具参数
    focus_on_the_primitive_character = request.json.get('focus_on_the_primitive_character')
    focus_on_the_path2 = request.json.get('focus_on_the_path2')
    # 错误原因显示参数
    cause = request.json.get('cause')
    # 治疗部位
    treatment = request.json.get('treatment')

    # 模型和数据
    model_id = request.json.get('model_id')
    data_id = request.json.get('data_id')
    model = Model.query.filter_by(id=model_id).first().model_url
    data = Data.query.filter_by(id=data_id).first().data_url
    model_name = Model.query.filter_by(id=model_id).first().model_name
    data_name = Data.query.filter_by(id=model_id).first().data_name
   # 第二次模型训练的预备参数
    result4 = os.system("sh ../model_doctor-main/scripts/train_model_doctor.sh " + model_name + " " + data_name)
    print("train_model_doctor.sh:", result4)

    '''
    此处利用以上参数进行训练，待补充
    '''
    # 训练结束后，保存权重以及保存任务记录
    now_time = datetime.now()
    now_time_str = datetime.strftime(now_time, '%Y%m%d%H%M%S')
    if not os.path.exists('{}{}'.format(WEIGHT_URL, now_time_str)):
        os.mkdir('{}{}'.format(WEIGHT_URL, now_time_str))  # 如果不存在权重url则构造权重url
    # 将权重文件保存到该文件夹(按时间戳分文件夹保存)
    #weight = np.arange(12)
    weight = os.path.join("../model_doctor-main/output", model_name + "_" + data_name + "_22051035", "models",
                          "model_optim.pth")
    '''
    待补充
    '''
    shutil.copyfile(weight, '{}{}/{}'.format(WEIGHT_URL, now_time_str, WEIGHT_NAME))
    #weight.tofile('{}{}/{}'.format(WEIGHT_URL, now_time_str, WEIGHT_NAME))
    # 新增任务记录
    task = Task(time=now_time_str, model_id=model_id, data_id=data_id,
                weight_url='{}{}/'.format(WEIGHT_URL, now_time_str))
    db.session.add(task)
    db.session.commit()  # 上传一条数据到task数据库中
    return redirect('/')


@app.route('/modeldoctor/task/result', methods=['get'])
def get_result():
    file_list = os.listdir('{}'.format(RESULT_URL))
    url_list = []
    for file in file_list:
        x = {
            "name": Path(file).stem,
            "url": url_for('static', _external=True, filename='result/{}'.format(file))
        }
        url_list.append(x)
    return url_list  # get诊断结果，从resulturl


@app.route('/modeldoctor/task/weight', methods=['get'])
def get_weight():
    late_finish_date = Task.query.order_by(db.desc(Task.time)).first().time
    late_finish_date = datetime.strftime(late_finish_date, '%Y%m%d%H%M%S')
    return send_from_directory('{}{}/'.format(WEIGHT_URL, late_finish_date), '{}'.format(WEIGHT_NAME),
                               as_attachment=True)


'''
    历史任务页面：
    1.进去时候刷新列表GET  /modeldoctor/history
    2.下载历史权重 GET /modeldoctor/history/<hid>
'''


@app.route('/modeldoctor/history', methods=['get'])
def get_history():
    task = Task.query.all()  # 查询Task表
    history = []
    for task_item in task:
        x = {
            "id": task_item.id,
            "time": task_item.time,
            "model": Model.query.filter_by(id=task_item.model_id).first().model_name,
            "data": Data.query.filter_by(id=task_item.data_id).first().data_name
        }
        history.append(x)
    return history  # get histor task表


@app.route('/modeldoctor/history/<hid>', methods=['get'])
def get_history_weight(hid):
    download_url = Task.query.filter_by(id=hid).first().weight_url  # 找到下载所需的url
    return send_from_directory(download_url, '{}'.format(WEIGHT_NAME), as_attachment=True)
  # 下载链接生成


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')  # 显示前端


if __name__ == '__main__':
    app.run(debug=True)
