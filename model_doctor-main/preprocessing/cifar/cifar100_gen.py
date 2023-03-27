import sys

sys.path.append('./')
sys.path.append('/home/lwd/Codes/modelOpt/modelopt-backend/model_doctor-main/')
import os
import pickle
import cv2
import numpy as np

from PIL import Image

# source directory

data_cifar100 = "model_doctor-main/datasets/cifar100"
CIFAR100_DIR = "model_doctor-main/datasets/cifar100/processed"

# extract cifar img in here.
CIFAR100_TRAIN_DIR = CIFAR100_DIR + '/train'
CIFAR100_TEST_DIR = CIFAR100_DIR + '/test'

dir_list = [CIFAR100_TRAIN_DIR, CIFAR100_TEST_DIR]


# extract the binaries, encoding must is 'bytes'!
def unpickle(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    return data


def gen_cifar_100():
    # generate training data sets.

    data_dir = data_cifar100 + '/train'
    train_data = unpickle(data_dir)
    print(data_dir + " is loading...")

    for i in range(0, 50000):
        img = np.reshape(train_data[b'data'][i], (3, 32, 32))

        r = img[0]
        g = img[1]
        b = img[2]

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
        img_path = CIFAR100_TRAIN_DIR + '/' + str(train_data[b'fine_labels'][i])
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = img_path + '/' + str(i) + '.png'
        rgb.save(filename, "PNG")
    print(data_dir + " loaded.")

    print("test_batch is loading...")

    # generate the validation data set.
    val_data = data_cifar100 + '/test'
    val_data = unpickle(val_data)
    for i in range(0, 10000):
        img = np.reshape(val_data[b'data'][i], (3, 32, 32))
        r = img[0]
        g = img[1]
        b = img[2]

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
        img_path = CIFAR100_TEST_DIR + '/' + str(train_data[b'fine_labels'][i])
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = img_path + '/' + str(i) + '.png'
        rgb.save(filename, "PNG")
    print("test_batch loaded.")
    return


if __name__ == '__main__':
    gen_cifar_100()