# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/8/26 10:06
# @Email：329201962@qq.com

import json
import os
import zipfile
from datetime import datetime
import time
from pathlib import Path
import shutil
import sys
import numpy as np
from flask import Flask, render_template, redirect, request, send_from_directory, url_for, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from common.getConfig import getConfig
from common.new_alchemy_encoder import new_alchemy_encoder

import io
from base64 import encodebytes
from PIL import Image
from flask import jsonify
####

app = Flask(__name__,
            template_folder="../modelopt-frontend/dist/",
            static_folder="../modelopt-frontend/dist/static/")

# app = Flask(__name__,
#             template_folder="../modelopt-frontend/dist/",
#             static_folder="../modelopt-frontend/dist/static/")

# 获取mysql配置信息
HOSTNAME = getConfig("config", 'mysql', 'host')  # def getConfig(filename, section, option):
PORT = getConfig("config", 'mysql', 'port')
USERNAME = getConfig("config", 'mysql', 'user')
PASSWORD = getConfig("config", 'mysql', 'password')
DATABASE = getConfig("config", 'mysql', 'db')


# 文件存储url配置信息
SCRIPT_URL = getConfig("config", 'save_url', 'scripts')
MODEL_URL = getConfig("config", "save_url", "model")
DATA_URL = getConfig("config", "save_url", "data")
OUTPUT_URL = getConfig("config", "save_url", "output")
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


class User(db.Model):
    __tablename__ = 'user'  # 表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rule = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)  # 四列数据构造

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



@app.route('/login', methods=['post'])
def login():
    print("login")
    check_name = request.json.get('name')
    check_password = request.json.get('password')
    user = User.query.filter_by(name=check_name).first()
    if user is None:
        return "0"
    password = user.password
    if password==check_password:
        return "1"
    else:
        return "0"

@app.route('/register', methods=['post'])
def register():
    print("register")
    register_name = request.json.get('name')
    register_password = request.json.get('password')
    register_email = request.json.get('email')
    print(register_name)
    print(User.query.filter_by(name=register_name))
    check_name = User.query.filter_by(name=register_name).first()
    
    if check_name is None:
        user = User(name=register_name, password=register_password, email=register_email, rule=0)
        db.session.add(user)
        db.session.commit()  
        return "1"
    else:
        return "0"        

@app.route('/motifyPassword', methods=['post'])
def motifyPassword():
    print("motifyPassword")
    name = request.json.get('name')
    old_password = request.json.get('old_password')
    new_password = request.json.get('new_password')
    user = User.query.filter_by(name=name).first()
    
    if user is not None and old_password==user.password:
        user.password = new_password
        db.session.add(user)
        db.session.commit()  
        return "1"
    else:
        return "0" 
    
# user management
@app.route('/getAllUsers', methods=['post'])
def getAllUsers():
    print("getAllUser")
    users = User.query.all()
    users = json.dumps(users, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    print(users)
    return users

@app.route('/addUser', methods=['post'])
def addUser():
    print("addUser")
    name = request.json.get('name')
    password = request.json.get('password')
    rule = request.json.get('rule')
    email = request.json.get('email')
    user = User.query.filter_by(name=name).first()
    if user is None:
        new_user = User(rule=rule, name=name, password=password, email=email)
        db.session.add(new_user)
        db.session.commit()  
        return "0"
    else:
        return "1" 

@app.route('/getSelectedusers', methods=['post'])
def getSelectedusers():
    print("getSelectedusers")
    name = request.json.get('name')
    users = User.query.filter(User.name.like('%'+name+'%')).all()
    users = json.dumps(users, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    print(users)
    return users

@app.route('/getEditUser', methods=['post'])
def getEditUser():
    print("getEditUser")
    name = request.json.get('name')
    user = User.query.filter_by(name=name).first()
    user = json.dumps(user, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    print(user)
    return user

@app.route('/submitEdit', methods=['post'])
def submitEdit():
    print("submitEdit")
    name = request.json.get('name')
    newpassword = request.json.get('newpassword')
    user = User.query.filter_by(name=name).first()
    user.password = newpassword
    db.session.commit()
    user = User.query.filter_by(name=name).first()
    if user.password == newpassword:
        return "0"
    else:
        return "1" 

@app.route('/deleteUser', methods=['post'])
def deleteUser():
    print("deleteUser")
    name = request.json.get('name')
    user = User.query.filter_by(name=name).first()
    db.session.delete(user)
    db.session.commit()  
    user = User.query.filter_by(name=name).first()
    if user is None: 
        return "0"
    else:
        return "1" 

@app.route('/delete_user', methods=['post'])
def delete_user():
    print("delete_user")
    name = request.json.get('name')
    user = User.query.filter_by(name=name).first()
    if user is not None:
        db.session.delete(user)
        db.session.commit()  
        return "1"
    else:
        return "0"    
    
       
'''
    模型页面
    1.进去刷新列表 GET /modeldoctor/model
    2.上传模型（检验模型是不是合法）POST /modeldoctor/model
'''


@app.route('/modeldoctor/model', methods=['get'])
def get_model_list():
    print("get_model_list")
    models = Model.query.all()  # 查询数据库所有项
    models = json.dumps(models, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    return models


@app.route('/modeldoctor/model', methods=['post'])
def add_model():
    print("add_model")
    file = request.files['file']
    model_name = file.filename.split(".")[0]
    print("model_name: ",model_name)
    assert model_name in ['simnet', 'alexnet', 'vgg16', 'resnet34', 'resnet50', 'senet34', 'wideresnet28', 'resnext50', 'densenet121', 'simplenetv1', 'efficientnetv2s', 'efficientnetv2l', 'googlenet', 'xception', 'mobilenetv2', 'inceptionv3', 'shufflenetv2', 'squeezenet', 'mnasnet']
    
    file.save('{}{}'.format(MODEL_URL, file.filename))
    model = Model(time=datetime.now(), model_name=model_name, model_url='{}{}'.format(MODEL_URL, file.filename))
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
    print("get_data_list")
    data = Data.query.all()  # 查询Data表
    data = json.dumps(data, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    return data  # 返回json数据


@app.route('/modeldoctor/dataset', methods=['post'])
def add_data():
    print("add_data")
    file = request.files['file']
    data_name = file.filename.split(".")[0]
    print("data_name: ",data_name)
    assert data_name in ['cifar10', 'cifar100', 'mnist', 'fashion-mnist', 'svhn', 'stl10', 'mini-imagenet']
    
    file.save('{}{}'.format(DATA_URL, file.filename))
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall(DATA_URL)
    zip_file.close()
    
    if data_name == 'cifar10':
        print("python ./model_doctor-main/preprocessing/cifar/"+data_name+"_gen.py")
        os.system("python ./model_doctor-main/preprocessing/cifar/"+data_name+"_gen.py")
        data_url="./model_doctor-main/datasets/cifar10"
    elif data_name == 'cifar100':
        print("python ./model_doctor-main/preprocessing/cifar/"+data_name+"_gen.py")
        os.system("python ./model_doctor-main/preprocessing/cifar/"+data_name+"_gen.py")
        data_url="./model_doctor-main/datasets/cifar100/processed"
    elif data_name == 'mnist':
        print("python ./model_doctor-main/preprocessing/mnist/image_gen.py")
        os.system("python ./model_doctor-main/preprocessing/mnist/image_gen.py")
        data_url="./model_doctor-main/datasets/mnist"
    elif data_name == 'fashion-mnist':
        print("python ./model_doctor-main/preprocessing/fashion-mnist/image_gen.py")
        os.system("python ./model_doctor-main/preprocessing/fashion-mnist/image_gen.py")
        data_url="./model_doctor-main/datasets/fashion-mnist"
    elif data_name == 'stl10':
        print("python ./model_doctor-main/preprocessing/stl10/image_gen_train.py")
        os.system("python ./model_doctor-main/preprocessing/stl10/image_gen_train.py")
        print("python ./model_doctor-main/preprocessing/stl10/image_gen_test.py")
        os.system("python ./model_doctor-main/preprocessing/stl10/image_gen_test.py")
        data_url="./model_doctor-main/datasets/stl10"
    elif data_name == 'mini-imagenet':
        print("python ./model_doctor-main/preprocessing/mini-imagenet/image_gen.py")
        os.system("python ./model_doctor-main/preprocessing/mini-imagenet/image_gen.py")
        data_url="./model_doctor-main/datasets/mini-imagenet/images"
    data = Data(time=datetime.now(), data_name=data_name, data_url=data_url)
    db.session.add(data)
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

@app.route('/modeldoctor/task/step1', methods=['post'])
def model_diagnose():
    # 模型和数据
    model_id = request.json.get('model_id')
    data_id = request.json.get('data_id')
    print("model_id:",model_id)
    print("data_id:",data_id)
    data_url = Data.query.filter_by(id=data_id).first().data_url####模型参数
    
    model_name = Model.query.filter_by(id=model_id).first().model_name
    data_name = Data.query.filter_by(id=data_id).first().data_name
    
    res_path = OUTPUT_URL + model_name + "_" + data_name
    
    if not os.path.exists(res_path+'/models/model_ori.pth'):
        print("sh " + SCRIPT_URL + "train.sh "+ model_name + " " + data_name + " " + data_url)
        result1=os.system("sh " + SCRIPT_URL + "train.sh " + model_name + " " + data_name + " " + data_url)
        print("result: ",result1)

    if not os.path.exists(res_path+"/images_50/"):
        print("sh " + SCRIPT_URL + "image_sift.sh " + model_name + " " + data_name + " " + data_url)
        result2 = os.system("sh " + SCRIPT_URL + "image_sift.sh " + model_name + " " + data_name + " " + data_url)
        print("result: ",result2)

    if not os.path.exists(res_path+"/grads_50/"):
        print("sh " + SCRIPT_URL + "grad_calculate.sh " + model_name + " " + data_name)
        result3 = os.system("sh " + SCRIPT_URL + "grad_calculate.sh " + model_name + " " + data_name)
        print("result: ",result3)

    if not os.path.exists(res_path+"/sift_visual/"):
        print("sh " + SCRIPT_URL + "grad_sift.sh " + model_name + " " + data_name + " " + data_url)
        result4 = os.system("sh " + SCRIPT_URL + "grad_sift.sh " + model_name + " " + data_name + " " + data_url)
        print("result: ",result4)

    if not os.path.exists(res_path+"/grad_visual/"):
        print("sh " + SCRIPT_URL + "grad_visualize.sh " + model_name + " " + data_name + " " + data_url)
        result5 = os.system("sh " + SCRIPT_URL + "grad_visualize.sh " + model_name + " " + data_name + " " + data_url)
        print("result: ",result5)
        
    if not os.path.exists(res_path+"/route_visual/route.jpg"):
        print("sh " + SCRIPT_URL + "model_route_path.sh " + model_name + " " + data_name + " " + data_url)
        result6 = os.system("sh " + SCRIPT_URL + "model_route_path.sh " + model_name + " " + data_name + " " + data_url)
        print("result: ",result6)

    
    # image_paths = [res_path+"/sift_visual/channel_grads_-1.png", res_path+"/grad_visual/grad response/high confidence/0.png"]
    # return jsonify({'result': image_paths})
    image_paths = {'channel':res_path+"/sift_visual/channel_grads_-1.png", 
                   'cam':res_path+"/grad_visual/grad response/high confidence/0.png", 
                   'origin':res_path+"/grad_visual/origin/0.png",
                   'route':res_path+"/route_visual/route.jpg"}
    encoded_imges = {}
    for type in image_paths:
        pil_img = Image.open(image_paths[type], mode='r') # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
        encoded_imges[type]=encoded_img
    return jsonify({'result': encoded_imges})


@app.route('/modeldoctor/task/pics', methods=['get'])
def get_pics():  # 返回图片，下载
    return send_from_directory('{}'.format(PICS_URL), '{}'.format(PICS_NAME), as_attachment=True)


@app.route('/modeldoctor/task/mark', methods=['post'])
def add_mark():  # 标注上传到MARK URL中（一个文件夹）
    file = request.files.get('mark')
    file.save('{}{}'.format(MARK_URL, file.filename))
    return redirect('/')


@app.route('/modeldoctor/task/step2', methods=['post'])
def model_treat():
    # 模型和数据
    model_id = request.json.get('model_id')
    data_id = request.json.get('data_id')
    kernel_radio = request.json.get('kernel_radio')
    background_radio = request.json.get('background_radio')
    route_radio = request.json.get('route_radio')
    layers = request.json.get('layers')
          
    data_url = Data.query.filter_by(id=data_id).first().data_url
    model_name = Model.query.filter_by(id=model_id).first().model_name
    data_name = Data.query.filter_by(id=data_id).first().data_name
    treat_model=1
    if kernel_radio == '1' and background_radio == '1':
        treat_model=1
    elif kernel_radio == '1':
        treat_model=2
    else:
        treat_model=3
    result1 = 1
    
    print("kernel_radio", kernel_radio)
    print("background_radio", background_radio)
    print("route_radio", route_radio)
    
    res_path = OUTPUT_URL + model_name + "_" + data_name
    
    if treat_model == 1:
        if not os.path.exists(res_path+"/models/model_optim_" + str(layers) + ".pth"):
            print("sh " + SCRIPT_URL + "train_model_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result1 = os.system("sh " + SCRIPT_URL + "train_model_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("train_model_doctor.sh:", result1)

        if not os.path.exists(res_path+"/images_50_optim_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "image_sift_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result2 = os.system("sh " + SCRIPT_URL + "image_sift_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result2)

        if not os.path.exists(res_path+"/grads_50_optim_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_calculate_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result3 = os.system("sh " + SCRIPT_URL + "grad_calculate_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result3)

        if not os.path.exists(res_path+"/sift_visual_optim_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_sift_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result4 = os.system("sh " + SCRIPT_URL + "grad_sift_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result4)

        if not os.path.exists(res_path+"/grad_visual_optim_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_visualize_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result5 = os.system("sh " + SCRIPT_URL + "grad_visualize_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result5)
            
        if not os.path.exists(res_path+"/route_visual/route_optim_" + str(layers) + ".jpg"):
            print("sh " + SCRIPT_URL + "model_route_path_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result6 = os.system("sh " + SCRIPT_URL + "model_route_path_doctor.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result6)

        
    elif treat_model == 2:
        if not os.path.exists(res_path+"/models/model_optim_spa_" + str(layers) + ".pth"):
            print("sh " + SCRIPT_URL + "train_model_doctor_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result1 = os.system("sh " + SCRIPT_URL + "train_model_doctor_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("train_model_doctor_spatial.sh:", result1)

        if not os.path.exists(res_path+"/images_50_optim_spa_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "image_sift_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result2 = os.system("sh " + SCRIPT_URL + "image_sift_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result2)

        if not os.path.exists(res_path+"/grads_50_optim_spa_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_calculate_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result3 = os.system("sh " + SCRIPT_URL + "grad_calculate_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result3)

        if not os.path.exists(res_path+"/sift_visual_optim_spa_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_sift_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result4 = os.system("sh " + SCRIPT_URL + "grad_sift_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result4)

        if not os.path.exists(res_path+"/grad_visual_optim_spa_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_visualize_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result5 = os.system("sh " + SCRIPT_URL + "grad_visualize_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result5)
            
        if not os.path.exists(res_path+"/route_visual/route_optim_spa_" + str(layers) + ".jpg"):
            print("sh " + SCRIPT_URL + "model_route_path_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result6 = os.system("sh " + SCRIPT_URL + "model_route_path_spatial.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result6)

        
    elif treat_model == 3:
        if not os.path.exists(res_path+"/models/model_optim_cha_" + str(layers) + ".pth"):
            print("sh " + SCRIPT_URL + "train_model_doctor_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result1 = os.system("sh " + SCRIPT_URL + "train_model_doctor_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("train_model_doctor_channel.sh:", result1)

        if not os.path.exists(res_path+"/images_50_optim_cha_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "image_sift_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result2 = os.system("sh " + SCRIPT_URL + "image_sift_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result2)

        if not os.path.exists(res_path+"/grads_50_optim_cha_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_calculate_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result3 = os.system("sh " + SCRIPT_URL + "grad_calculate_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result3)

        if not os.path.exists(res_path+"/sift_visual_optim_cha_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_sift_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result4 = os.system("sh " + SCRIPT_URL + "grad_sift_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result4)

        if not os.path.exists(res_path+"/grad_visual_optim_cha_" + str(layers) + "/"):
            print("sh " + SCRIPT_URL + "grad_visualize_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result5 = os.system("sh " + SCRIPT_URL + "grad_visualize_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result5)
            
        if not os.path.exists(res_path+"/route_visual/route_optim_cha_" + str(layers) + ".jpg"):
            print("sh " + SCRIPT_URL + "train_model_doctor_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            result6 = os.system("sh " + SCRIPT_URL + "train_model_doctor_channel.sh " + model_name + " " + data_name + " " + data_url + " " + str(layers))
            print("result: ",result6)

    # 训练结束后，保存权重以及保存任务记录
    now_time = datetime.now()
    now_time_str = datetime.strftime(now_time, '%Y%m%d%H%M%S')
    if not os.path.exists('{}{}'.format(WEIGHT_URL, now_time_str)):
        os.mkdir('{}{}'.format(WEIGHT_URL, now_time_str))  # 如果不存在权重url则构造权重url
    # 将权重文件保存到该文件夹(按时间戳分文件夹保存)
    if treat_model == 1:
        weight = os.path.join("./model_doctor-main/output", model_name + "_" + data_name , "models",
                            'model_optim_' + str(layers) + '.pth')
    elif treat_model == 2:
        weight = os.path.join("./model_doctor-main/output", model_name + "_" + data_name , "models",
                            'model_optim_spa_' + str(layers) + '.pth')
    elif treat_model == 3:
        weight = os.path.join("./model_doctor-main/output", model_name + "_" + data_name , "models",
                            'model_optim_cha_' + str(layers) + '.pth')
        
    shutil.copyfile(weight, '{}{}/{}'.format(WEIGHT_URL, now_time_str, WEIGHT_NAME))
    # 新增任务记录
    task = Task(time=now_time_str, model_id=model_id, data_id=data_id,
                weight_url='{}{}/'.format(WEIGHT_URL, now_time_str))
    db.session.add(task)
    db.session.commit()  # 上传一条数据到task数据库中
    # return redirect('/')
    if treat_model == 1:
        image_paths = {'channel':res_path+"/sift_visual_optim_" + str(layers) + "/channel_grads_-1.png", 
                    'cam':res_path+"/grad_visual_optim_" + str(layers) + "/grad response/high confidence/0.png", 
                    'origin':res_path+"/grad_visual_optim_" + str(layers) + "/origin/0.png",
                    'route':res_path+"/route_visual/route_optim_" + str(layers) + ".jpg"}
    elif treat_model == 2:
        image_paths = {'channel':res_path+"/sift_visual_optim_spa_" + str(layers) + "/channel_grads_-1.png", 
                    'cam':res_path+"/grad_visual_optim_spa_" + str(layers) + "/grad response/high confidence/0.png", 
                    'origin':res_path+"/grad_visual_optim_spa_" + str(layers) + "/origin/0.png",
                    'route':res_path+"/route_visual/route_optim_spa_" + str(layers) + ".jpg"}
    elif treat_model == 3:
        image_paths = {'channel':res_path+"/sift_visual_optim_cha_" + str(layers) + "/channel_grads_-1.png", 
                        'cam':res_path+"/grad_visual_optim_cha_" + str(layers) + "/grad response/high confidence/0.png", 
                        'origin':res_path+"/grad_visual_optim_cha_" + str(layers) + "/origin/0.png",
                        'route':res_path+"/route_visual/route_optim_cha_" + str(layers) + ".jpg"}
            
    encoded_imges = {}
    for type in image_paths:
        pil_img = Image.open(image_paths[type], mode='r') # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
        encoded_imges[type]=encoded_img
    return jsonify({'result': encoded_imges})


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


@app.route('/modeldoctor/task/weight', methods=['post'])
def get_weight():
    late_finish_date = Task.query.order_by(db.desc(Task.time)).first().time
    late_finish_date = datetime.strftime(late_finish_date, '%Y%m%d%H%M%S')
    print('{}{}/'.format(WEIGHT_URL, late_finish_date), '{}'.format(WEIGHT_NAME))
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
    print("get_history")
    for task_item in task:
        x = {
            "id": task_item.id,
            "date": str(task_item.time),
            "model": Model.query.filter_by(id=task_item.model_id).first().model_name,
            "data": Data.query.filter_by(id=task_item.data_id).first().data_name
        }
        history.append(x)
    history = json.dumps(history, cls=new_alchemy_encoder(), check_circular=False)  # 进行json序列化
    return history  # 返回json数据


@app.route('/modeldoctor/history/hid', methods=['get'])
def get_history_weight():
    hid = request.args.get('id')
    print(Task.query.filter_by(id=hid))
    download_url = Task.query.filter_by(id=hid).first().weight_url  # 找到下载所需的url
    
    try:
        response = make_response(
            send_from_directory(download_url, '{}'.format(WEIGHT_NAME), as_attachment=True))
        return response
    except Exception as e:
        return jsonify({"code": "异常", "message": "{}".format(e)})


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')  # 显示前端


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444,debug=True)