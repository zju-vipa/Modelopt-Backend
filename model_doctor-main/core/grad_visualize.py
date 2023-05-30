import sys


sys.path.append('/home/lwd/Codes/modelOpt/modelopt-backend/model_doctor-main/')

import argparse
# from configs import config
import os
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Grayscale
import seaborn as sns
import matplotlib.pyplot as plt
import json

import PIL.Image as Image
import loaders
import models


def save_origin(save_path, save_name, images):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = '{}/{}.png'.format(save_path, save_name)
    images.save(filename)
    # cv2.imwrite(filename, images)

        
    
def save_heatmap(save_path, save_name, heatmap, is_whole=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_whole:
        filename = '{}/{}.png'.format(save_path, save_name)
        cv2.imwrite(filename, heatmap)
    else:
        for i, m in enumerate(heatmap):
            filename = '{}/{}_{}.png'.format(save_path, save_name, i)
            cv2.imwrite(filename, m)


def gen_heatmap(grad, size, img=None, is_whole=True):
    grad = torch.abs(grad)  # grad processing
    grad = grad.detach().numpy()[0, :]
    if is_whole:
        grad = np.sum(grad, axis=0)  # whole -> sum
        grad = grad - np.min(grad)
        grad = grad / np.max(grad)
        # grad = cv2.resize(grad, (224, 224))
        grad = cv2.resize(grad, (size, size))##cifar10

        heatmap = cv2.applyColorMap(np.uint8(255 * grad), cv2.COLORMAP_JET)  # gen heatmap
        if img is not None:
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap + np.float32(img)
            heatmap = heatmap / np.max(heatmap)
            heatmap = np.uint8(255 * heatmap)
        else:
            return heatmap
        return heatmap
    else:
        heatmaps = np.zeros((grad.shape[0], 224, 224, 3), dtype=np.float32)  # part -> inter
        grad_min = np.min(grad)
        grad_max = np.max(grad)
        for i, g in enumerate(grad):
            g = g - grad_min
            g = g / grad_max
            g = cv2.resize(g, (224, 224))

            heatmap = cv2.applyColorMap(np.uint8(255 * g), cv2.COLORMAP_JET)
            if img is not None:
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap + np.float32(img)
                heatmap = heatmap / np.max(heatmap)
                heatmaps[i] = np.uint8(255 * heatmap)
            else:
                heatmaps[i] = heatmap
        return heatmaps

def _img_loader(path, mode='RGB'):
    assert mode in ['RGB', 'L']
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def image_process(image_path):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])(rgb_img).unsqueeze(0)
    return input_tensor, rgb_img


def image_process_cifar10(image_path,transform):
    image = _img_loader(image_path, mode='RGB')
    # name = os.path.split(image_path)[1]
    # transform = Compose([
    #                 ToTensor(),
    #                 Normalize((0.4914, 0.4822, 0.4465),
    #                             (0.2023, 0.1994, 0.2010))
    #             ])
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, image


def label_process(class_name,class_dic):
    # class_dic = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    #              'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9} # cifar
    # class_dic = {'n01770081': 0, 'n02091831': 1, 'n02108089': 2, 'n02687172': 3, 'n04251144': 4,
    #              'n04389033': 5, 'n04435653': 6, 'n04443257': 7, 'n04515003': 8, 'n07747607': 9}
    label = torch.tensor(class_dic[class_name]).unsqueeze(0)
    return label


class HookModule:
    def __init__(self, model, module):
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        self.model.zero_grad()
        return grads

def calculate_grad(model, module, labels, inputs):
    module = HookModule(model=model, module=module)

    outputs = model(inputs)
    softmax = torch.nn.Softmax(dim=1)(outputs)
    scores, predicts = torch.max(softmax, 1)
    print('=== forward ===>', predicts, scores)

    nll_loss = torch.nn.NLLLoss()(outputs, labels)
    grads = module.grads(outputs=-nll_loss, inputs=module.activations,
                         retain_graph=True, create_graph=False)
    nll_loss.backward()  # to release graph

    return grads


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default=3, type=int, help='in channels')
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--result_path', default='', type=str, help='image path')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    np.set_printoptions(threshold=np.inf)
    # for debug
    # args.data_name = 'stl10'
    # args.model_name = 'alexnet'

    # args.model_path = os.path.join('model_doctor-main/output/', args.model_name+'_'+args.data_name , 'models/model_ori.pth')
    # args.result_path = os.path.join('model_doctor-main/output/', args.model_name+'_'+args.data_name, 'grad_visual')
    # args.data_path = os.path.join('model_doctor-main/datasets/',args.data_name, 'test')
    # print(args.model_path)
    # print(args.data_path)
    # print(args.result_path)
    # for  debug
    
    cfg = json.load(open('model_doctor-main/configs/config_trainer.json'))[args.data_name]
    
    args.in_channels=cfg['model']['in_channels']
    args.num_classes=cfg['model']['num_classes']
    
    model_name = args.model_name
    data_name = args.data_name
    in_channels=args.in_channels
    num_classes=args.num_classes
    
    data_path = args.data_path
    
    model_path = args.model_path
    result_path = args.result_path
    
    print(model_path)
    print(data_path)
    print(result_path)
    
    # in_channels=3
    # num_classes=10

    model_layers = [-1]
    # model = models.load_model(model_name=model_name,
    #                           in_channels=in_channels,
    #                           num_classes=num_classes)
    model = models.load_model(model_name=args.model_name, data_name=args.data_name, in_channels=args.in_channels, num_classes=args.num_classes)
    # print(model)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    module = models.load_modules(model=model,
                                 model_layers=model_layers)[0]

    if data_name == 'cifar10' :
        test_images = {
            'airplane': '3.png',
            'automobile': '6.png',
            'bird': '25.png',
            'cat': '0.png',
            'deer': '22.png',
            'dog': '12.png',
            'frog': '4.png',
            'horse': '13.png',
            'ship': '1.png',
            'truck': '11.png',
        }
        transform = Compose([
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
            ])
        size=32
        class_dic = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9} # cifar
    elif data_name == 'mini-imagenet':
        test_images = {
            'n01770081': 'n0177008100000075.jpg',
            'n02091831': 'n0209183100000171.jpg',
            'n02108089': 'n0210808900000206.jpg',
            'n02687172': 'n0268717200000050.jpg',
            'n04251144': 'n0425114400000079.jpg',
            'n04389033': 'n0438903300000100.jpg',
            'n04435653': 'n0443565300000079.jpg',
            'n04443257': 'n0444325700000038.jpg',
            'n04515003': 'n0451500300000148.jpg',
            'n07747607': 'n0774760700000034.jpg',
        }
        transform = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
            ])
        size=224
        class_dic = {'n01770081': 0, 'n02091831': 1, 'n02108089': 2, 'n02687172': 3, 'n04251144': 4,
                    'n04389033': 5, 'n04435653': 6, 'n04443257': 7, 'n04515003': 8, 'n07747607': 9}
    elif data_name == 'stl10':
        test_images = {
            '1': '3.png',
            '2': '5.png',
            '3': '24.png',
            '4': '4.png',
            '5': '8.png',
            '6': '2.png',
            '7': '0.png',
            '8': '1.png',
            '9': '55.png',
            '10': '17.png',
        }
        transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
            ])
        size=96
        class_dic = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                    '6': 5, '7': 6, '8': 7, '9': 8, '10': 9}
    elif data_name == 'mnist':
        test_images = {
            '0': 'test_3.jpg',
            '1': 'test_2.jpg',
            '2': 'test_1.jpg',
            '3': 'test_18.jpg',
            '4': 'test_4.jpg',
            '5': 'test_8.jpg',
            '6': 'test_11.jpg',
            '7': 'test_0.jpg',
            '8': 'test_61.jpg',
            '9': 'test_7.jpg'
        }
        transform = Compose([
                Resize((32, 32)),
                Grayscale(1),
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
        size=28
        class_dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    elif data_name == 'fashion-mnist':
        test_images = {
            '0': 'test_19.jpg',
            '1': 'test_2.jpg',
            '2': 'test_1.jpg',
            '3': 'test_13.jpg',
            '4': 'test_6.jpg',
            '5': 'test_8.jpg',
            '6': 'test_4.jpg',
            '7': 'test_9.jpg',
            '8': 'test_18.jpg',
            '9': 'test_0.jpg'
        }
        transform = Compose([
                Resize((32, 32)),
                Grayscale(1),
                ToTensor(),
                Normalize((0.286,), (0.353,))
            ])
        size=28
        class_dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    elif data_name == 'cifar100':
        test_images = {
            '0': '9.png',
            '1': '132.png',
            '2': '54.png',
            '3': '396.png',
            '4': '50.png',
            '5': '334.png',
            '6': '51.png',
            '7': '64.png',
            '8': '27.png',
            '9': '52.png'
        }
        transform = Compose([
                ToTensor(),
                Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                          (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ])
        size=32
        class_dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
          
    for i, name in enumerate(test_images):
        print('-' * 40)
        class_name = name
        image_name = test_images[name]
        save_path = os.path.join(result_path, 'grad response', 'high confidence')
        origin_path = os.path.join(result_path, 'origin')
        image_path = '{}/{}/{}'.format(data_path, class_name, image_name)

        labels = label_process(class_name, class_dic)
        inputs, images = image_process_cifar10(image_path,transform)

        grads = calculate_grad(model, module, labels, inputs)
        
        heatmap = gen_heatmap(grad=grads,size=size, img=images, is_whole=True)
        save_heatmap(save_path=save_path, save_name=i, heatmap=heatmap, is_whole=True)
        save_origin(save_path=origin_path , save_name=i, images=images)

    # main() #only main
