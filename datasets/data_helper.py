import os
import random
from lib.utils.utils import *

DATA_ROOT = r'/home/hanhu/Spoof_Croped'

TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Val/'
RESIZE_SIZE = 256

def load_train_list(root_folder, list_file):
    list = []
    f = open(os.path.join(root_folder, list_file))
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list(root_folder, list_file):
    list = []
    f = open(os.path.join(root_folder, list_file))
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list(root_folder, list_file):
    list = []
    f = open(os.path.join(root_folder, list_file))
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[1]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]




