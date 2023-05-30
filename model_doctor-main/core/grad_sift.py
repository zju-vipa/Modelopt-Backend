import sys


sys.path.append('/home/lwd/Codes/modelOpt/modelopt-backend/model_doctor-main/')

import argparse
import os
import numpy as np
import torch
import json

import loaders
import models
# from configs import config
from utils import image_util



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


class GradSift:
    def __init__(self, class_nums, grad_nums):
        self.class_nums = class_nums
        self.grad_nums = grad_nums
        self.grads = None
        self.scores = torch.zeros((class_nums, grad_nums))
        self.nums = torch.zeros(class_nums, dtype=torch.long)

    def __call__(self, outputs, labels, grads):
        if self.grads is None:
            self.grads = torch.zeros((self.class_nums,
                                      self.grad_nums,
                                      grads.shape[1],
                                      grads.shape[2],
                                      grads.shape[3]))
            print(self.grads.shape)

        softmax = torch.softmax(outputs, dim=1)
        scores, predicts = torch.max(softmax, dim=1)
        for i, label in enumerate(labels):
            if label == predicts[i]:
                if self.nums[label] == self.grad_nums:
                    score_min, index = torch.min(self.scores[label], dim=0)
                    if scores[i] > score_min:
                        self.scores[label][index] = scores[i]
                        self.grads[label][index] = grads[i]
                else:
                    self.scores[label][self.nums[label]] = scores[i]
                    self.grads[label][self.nums[label]] = grads[i]
                    self.nums[label] += 1

    def sum_channel(self, result_path, model_layer):
        print(self.scores)
        print(self.nums)

        grads = torch.abs(self.grads)
        view_channel(grads, result_path, model_layer)
        # grads_pos = torch.nn.ReLU()(self.grads)
        # grads_neg = torch.nn.ReLU()(-self.grads)


def view_channel(grads, result_path, model_layer):
    # grads numpy
    grads_sum = torch.sum(grads, dim=(1, 3, 4)).detach().numpy()
    grads_path = os.path.join(result_path, 'channel_grads_{}.npy'.format(model_layer))
    np.save(grads_path, grads_sum)

    # grads numpy view
    grads_path = os.path.join(result_path, 'channel_grads_{}.png'.format(model_layer))
    image_util.view_grads(grads_sum, 512, 10, grads_path)

    # # grads numpy sort view
    # grads_sum_sort = -np.sort(-grads_sum, axis=1)
    # grads_path = os.path.join(result_path, 'channel_grads_{}_sort.png'.format(model_layer))
    # image_util.view_grads(grads_sum_sort, 512, 10, grads_path)
    #
    # sift_channel(result_path, model_layer)


def sift_channel(result_path, model_layer, threshold=None):  # high response channel
    grads_path = os.path.join(result_path, 'channel_grads_{}.npy'.format(model_layer))
    channels_grads = np.load(grads_path)

    if threshold is None:
        channels_threshold = channels_grads.mean(axis=1)
    else:
        channels_threshold = -np.sort(-channels_grads, axis=1)[:, threshold]  # -sort从小到大

    channels = np.ones(shape=channels_grads.shape)
    for c, t in enumerate(channels_threshold):
        channels[c] = np.where(channels_grads[c] >= t, 1, 0)

    channel_path = os.path.join(result_path, 'channels_{}.npy'.format(model_layer))
    np.save(channel_path, channels)

    print(channels)
    print(channels_threshold)


# ----------------------------------------
# test
# ----------------------------------------
def sift_grad(data_name, model_name, model_layers, model_path, result_path):
    # device
    device = torch.device('cuda:0')
    
    model = models.load_model(model_name=model_name, data_name=data_name, in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # data
    train_loader = loaders.load_data(data_name=data_name,data_path=data_path ,data_type='train')

    # grad
    module = HookModule(model=model, module=models.load_modules(model, model_layers)[0])
    # module = HookModule(model=model, module=models.load_modules(model, model_name, model_layers)[0])
    grad_sift = GradSift(class_nums=num_classes, grad_nums=100)
    # grad_sift = GradSift(class_nums=cfg['model']['num_classes'], grad_nums=100)

    # forward
    for i, samples in enumerate(train_loader):
        print('\r[{}/{}]'.format(i, len(train_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        grads = module.grads(outputs=-nll_loss, inputs=module.activations,
                             retain_graph=True, create_graph=False)
        nll_loss.backward()  # to release graph

        grad_sift(outputs=outputs, labels=labels, grads=grads)

    print('\n', end='', flush=True)
    grad_sift.sum_channel(result_path, model_layers[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default='', type=int, help='in channels')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--result_path', default='', type=str, help='image path')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    np.set_printoptions(threshold=np.inf)
    
    # model_name = 'alexnet'
    # data_name = 'cifar10'
    # in_channels=3
    # num_classes=10
    
    # data_path = os.path.join('model_doctor-main/datasets/',data_name, 'test')
    
    # model_path = os.path.join('model_doctor-main/output/', model_name+'_'+data_name , 'models/model_ori.pth')
    # result_path = os.path.join('model_doctor-main/output/', model_name+'_'+data_name, 'sift_visual')
    
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




    model_layers = [-1]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    sift_grad(data_name, model_name, model_layers, model_path, result_path)

