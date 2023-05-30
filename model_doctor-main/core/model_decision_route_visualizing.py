from tkinter import *
import numpy as np
import os

from PIL import Image, ImageDraw

import sys

sys.path.append('.')
sys.path.append('/home/lwd/Codes/modelOpt/modelopt-backend/model_doctor-main/')

import argparse
import os
import json
from tqdm import tqdm

import math


CONV_W = 4
CONV_H = 4
LINEAR_W = 2
LINEAR_H = 2

INTERVAL_CONV_X = 200
INTERVAL_CONV_Y = 7
INTERVAL_LINEAR_X = 280
INTERVAL_LINEAR_Y = 4.5

PADDING_X = 10
PADDING_Y = 2500  # middle line

LINE_WIDTH = 1

# COLOR_PUBLIC = 'orange'
# COLOR_NO_USE = 'gray'
# COLORS = ['purple', 'red']
# COLOR_PUBLIC = '#feb888'
# COLOR_NO_USE = '#c8c8c8'
# COLORS = ['#b0d994', '#a3cbef', ]
COLOR_PUBLIC = 'orange'
COLOR_NO_USE = 'purple'
COLORS = ['#C82423', '#2878B5', ]
COLORS1 = ['gray', 'red']
# COLORS = ['#2878B5', '#C82423', ]
white = (222, 222, 220)
def arrowedLine(im, ptA, ptB, width=1, color=(0,255,0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    draw = ImageDraw.Draw(im)
    # Draw the line without arrows
    draw.line((ptA,ptB), width=width, fill=color)

    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.95*(x1-x0)+x0
    yb = 0.95*(y1-y0)+y0

    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0==x1:
       vtx0 = (xb-5, yb)
       vtx1 = (xb+5, yb)
    # Check if line is horizontal
    elif y0==y1:
       vtx0 = (xb, yb+5)
       vtx1 = (xb, yb-5)
    else:
       alpha = math.atan2(y1-y0,x1-x0)-90*math.pi/180
       a = 8*math.cos(alpha)
       b = 8*math.sin(alpha)
       vtx0 = (xb+a, yb+b)
       vtx1 = (xb-a, yb-b)

    #draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line

    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im

def draw_route(masks, layers, result_path, result_name, FIG_W, FIG_H):
    PADDING_Y = int(FIG_H/2)
    image1 = Image.new("RGB", (FIG_W, FIG_H), white)
    draw = ImageDraw.Draw(image1)
    #  ---------------------------
    #  each layer
    #  ---------------------------
    masks = np.asarray(masks)  # layers, labels, channels
    print(masks.shape)

    x = PADDING_X
    line_start_p_preceding = [(PADDING_X, PADDING_Y)]  # public
    line_start_preceding = [[(PADDING_X, PADDING_Y)] for _ in range(masks.shape[1])]  # [labels * [init]]

    for layer in range(masks.shape[0]):

        line_end_p = []  # public
        line_start_p = []  # public
        line_end = [[] for _ in range(masks.shape[1])]  # [labels * []] each class
        line_start = [[] for _ in range(masks.shape[1])]

        line_p_num = 0
        line_num = 0

        #  ---------------------------
        #  each channel
        #  ---------------------------
        layer_masks = np.asarray(list(masks[layer]))  # labels, channels
        print(layer)
        # init posi.
        if layers[layer] == 'conv':
            x += CONV_W + INTERVAL_CONV_X
            y = PADDING_Y - (layer_masks.shape[1] / 2) * (CONV_H + INTERVAL_CONV_Y) + INTERVAL_CONV_Y / 2
        else:
            x += LINEAR_W + INTERVAL_LINEAR_X
            y = PADDING_Y - (layer_masks.shape[1] / 2) * (LINEAR_H + INTERVAL_LINEAR_Y) + INTERVAL_LINEAR_Y / 2

        # draw conv/linear
        for channel in range(layer_masks.shape[1]):
            if layer_masks[:, channel].sum() > 1:
                if layers[layer] == 'conv':
                    line_end_p.append(((x), (y + CONV_H / 2)))
                    line_start_p.append(((x + CONV_W), (y + CONV_H / 2)))
                    draw.rectangle((x, y, x + CONV_W, y + CONV_H),
                                        outline=COLOR_PUBLIC,
                                        fill=COLOR_PUBLIC,
                                        width=LINE_WIDTH)
                else:
                    line_end_p.append(((x), (y + LINEAR_H / 2)))
                    line_start_p.append(((x + LINEAR_W), (y + LINEAR_H / 2)))
                    draw.ellipse((x, y, x + LINEAR_W, y + LINEAR_H),
                                   outline=COLOR_PUBLIC,
                                   fill=COLOR_PUBLIC,
                                   width=LINE_WIDTH)
            elif layer_masks[:, channel].sum() < 1:
                if layers[layer] == 'conv':
                    draw.rectangle((x, y, x + CONV_W, y + CONV_H),
                                        outline=COLOR_NO_USE,
                                        fill=COLOR_NO_USE,
                                        width=LINE_WIDTH)
                else:
                    draw.ellipse((x, y, x + LINEAR_W, y + LINEAR_H),
                                   outline=COLOR_NO_USE,
                                   fill=COLOR_NO_USE,
                                   width=LINE_WIDTH)
            else:
                #  ---------------------------
                #  each label
                #  ---------------------------
                for l, mask in enumerate(layer_masks[:, channel]):
                    if mask:
                        if layers[layer] == 'conv':
                            line_end[l].append(((x), (y + CONV_H / 2)))
                            line_start[l].append(((x + CONV_W), (y + CONV_H / 2)))
                            # print(COLORS1[l])
                            draw.rectangle((x, y, x + CONV_W, y + CONV_H),outline=COLORS1[l],fill=COLORS1[l],width=LINE_WIDTH)
                        else:
                            line_end[l].append(((x), (y + LINEAR_H / 2)))
                            line_start[l].append(((x + LINEAR_W), (y + LINEAR_H / 2)))
                            draw.ellipse((x, y, x + LINEAR_W, y + LINEAR_H),
                                           outline=COLORS1[l],
                                           fill=COLORS1[l],
                                           width=LINE_WIDTH)

            # next y start posi.
            if layers[layer] == 'conv':
                y += CONV_H + INTERVAL_CONV_Y
            else:
                y += LINEAR_H + INTERVAL_LINEAR_Y

        # draw line
        for l in range(layer_masks.shape[0]):
            # line_num += (len(line_start_preceding[l]) * len(line_end[l]))  # each to each
            # line_p_num += (len(line_start_preceding[l]) * len(line_end_p))  # each to public
            # line_p_num += (len(line_start_p_preceding) * len(line_end[l]))  # public to each
            line_num += len(line_start[l])  # each
            for x0, y0 in line_start_preceding[l]:
                # each to each
                for x1, y1 in line_end[l]:
                    image1 = arrowedLine(image1,(x0, y0),(x1, y1),width=LINE_WIDTH,color=COLORS1[l])
                    draw = ImageDraw.Draw(image1)
                    # draw.line((x0, y0, x1, y1),
                    #                width=LINE_WIDTH,
                    #                fill=COLORS1[l],
                    #                # arrow=LAST,
                    #                arrowshape=(6, 5, 1))

                # each to public
                for x1, y1 in line_end_p:
                    image1 = arrowedLine(image1,(x0, y0),(x1, y1),width=LINE_WIDTH,color=COLORS1[l])
                    draw = ImageDraw.Draw(image1)
                    # draw.line((x0, y0, x1, y1),
                    #                width=LINE_WIDTH,
                    #                fill=COLORS1[l],
                    #                # arrow=LAST,
                    #                arrowshape=(6, 5, 1))

            # public to each
            for x0, y0 in line_start_p_preceding:
                for x1, y1 in line_end[l]:
                    image1 = arrowedLine(image1,(x0, y0),(x1, y1),width=LINE_WIDTH,color=COLORS1[l])
                    draw = ImageDraw.Draw(image1)
                    # draw.line((x0, y0, x1, y1),
                    #                width=LINE_WIDTH,
                    #                fill=COLORS1[l],
                    #                # arrow=LAST,
                    #                arrowshape=(6, 5, 1))

        # line_p_num += (len(line_start_p_preceding) * len(line_end_p))  # public to public
        line_p_num += len(line_start_p)  # public
        # public to public
        for x0, y0 in line_start_p_preceding:
            for x1, y1 in line_end_p:
                image1 = arrowedLine(image1,(x0, y0),(x1, y1),width=LINE_WIDTH,color=COLORS1[l])
                draw = ImageDraw.Draw(image1)
                # draw.line((x0, y0, x1, y1),
                #                width=LINE_WIDTH + 1,
                #                fill=COLOR_PUBLIC,
                #                # arrow=LAST,
                #                arrowshape=(6, 5, 1))

        line_start_preceding = line_start.copy()
        line_start_p_preceding = line_start_p.copy()

        # calculate
        print('--->', layer,
              '| line--->', line_num,
              '| line_p--->', line_p_num,
              '| --->', line_p_num / (line_num + line_p_num))
    # filename = "my_drawing.jpg"
    
    image1.save(result_path+'/'+result_name)


def load_mask(grad_path,result_path,model_name,data_name,in_channels,num_classes):
    if model_name == 'alexnet':
        labels = [2]
        layers = [4, 3, 2, 1, 0]  # inputs
        layers_name = ['conv' for _ in range(5)] + ['linear' for _ in range(3)]
        layer_masks1 = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        FIG_W = 1536
        FIG_H = 4504
        print(num_classes)
    elif  model_name == 'vgg16':
        labels = [2]
        layers = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # inputs
        layers_name = ['conv' for _ in range(13)] + ['linear' for _ in range(3)]
        layer_masks1 = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        FIG_W = 4236
        FIG_H = 8024
        print(num_classes)
    elif  model_name == 'simnet':
        labels = [2]
        layers = [2, 1, 0]  # inputs
        layers_name = ['conv' for _ in range(3)] + ['linear' for _ in range(1)]
        layer_masks1 = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        FIG_W = 1026
        FIG_H = 1024
        print(num_classes)
    masks = []
    for layer in layers:
        layer_masks = []
        for label in labels:
            output_channel = np.load(f'{grad_path}/layer_{layer}.npy')[label]
            layer_masks.append(output_channel)
        masks.append(layer_masks)
        print(np.asarray(layer_masks).shape)

    masks.append(layer_masks1)
    return masks, layers_name, FIG_W, FIG_H


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default='', type=int, help='in channels')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--grad_path', default='', type=str, help='grad path')
    parser.add_argument('--result_path', default='', type=str, help='image path')
    parser.add_argument('--result_name', default='', type=str, help='result name')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    np.set_printoptions(threshold=np.inf)
    
    # model_name = 'alexnet'
    # data_name = 'cifar10'
    # in_channels=3
    # num_classes=10
    
    
    cfg = json.load(open('model_doctor-main/configs/config_trainer.json'))[args.data_name]
    
    args.in_channels=cfg['model']['in_channels']
    args.num_classes=cfg['model']['num_classes']
    
    
    model_name = args.model_name
    data_name = args.data_name
    in_channels=args.in_channels
    num_classes=args.num_classes
    
    # data_path = args.data_path
    
    grad_path = args.grad_path
    result_path = args.result_path
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    print(grad_path)
    # print(data_path)
    print(result_path)

    FIG_W = 1536
    FIG_H = 5024

    masks, layers, FIG_W, FIG_H = load_mask(grad_path,result_path, model_name,data_name,in_channels,num_classes)
    # print(np.asarray(masks).shape)
    print(len(masks))
    print(len(masks[0]))
    print(len(masks[0][0]))
    print(layers)

    draw_route(masks, layers, result_path, args.result_name, FIG_W, FIG_H)


if __name__ == '__main__':
    

    main()
