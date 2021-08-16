import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random

def default_loader(path):
    # BGR
    return Image.open(path).convert('RGB')

class PairDataLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader=default_loader, stage='Train'):
        self.loader = loader
        self.transform = transform
        self.root_folder = root_folder

        fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
        label_dict = dict()
        for file_name, label in fp_items:
            label = int(label)
            if not label_dict.has_key(label):
                label_dict[label] = []
            label_dict[label].append(file_name.strip())

        imgs_count = []
        print('\nStage: ' + stage)
        for key in label_dict.keys():
            imgs_count.append(len(label_dict[key]))
            if len(label_dict[key]) < 2:
                raise Exception('The samples of {} less than two.'.format(key))
            random.shuffle(label_dict[key])
        imgs_count = np.array(imgs_count)

        if len(label_dict) != (max(label_dict.keys()) + 1):
            raise Exception('The label of filename must from 0 to N-1.')
        
        self.label_dict = label_dict
        self.label_count = len(self.label_dict)
        self.fp_items = fp_items

        print('The distinct label number: {}. The number of imgs: {}.'.format(len(label_dict), len(fp_items)))
        print('imgs per class statistics: min({}), max({}), mean({}).'.format(imgs_count.min(), imgs_count.max(), imgs_count.mean()))
                

    def __getitem__(self, index):
        
        filenames = self.label_dict[index]
        idx1 = np.random.choice(len(filenames))
        idx2 = np.random.choice(len(filenames))
        while idx2 == idx1:
            idx2 = np.random.choice(len(filenames))
        filename_p1, filename_p2 = filenames[idx1], filenames[idx2]

        img_p1 = self.loader(os.path.join(self.root_folder, filename_p1))
        img_p2 = self.loader(os.path.join(self.root_folder, filename_p2))
        if self.transform is not None:
            img_p1 = self.transform(img_p1)
            img_p2 = self.transform(img_p2)

        return img_p1, img_p2, index

    def __len__(self):
        return self.label_count

