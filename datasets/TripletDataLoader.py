import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random

def default_loader(path):
    # BGR
    return Image.open(path).convert('RGB')

class TripletDataLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader=default_loader):
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
        for key in label_dict.keys():
            imgs_count.append(len(label_dict[key]))
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
        anchor_filename, anchor_label = self.fp_items[index]
        anchor_label = int(anchor_label)

        positive_idx = np.random.choice(len(self.label_dict[anchor_label]))
        while self.label_dict[anchor_label][positive_idx] == anchor_filename:
            positive_idx = np.random.choice(len(self.label_dict[anchor_label]))
            
        positive_filename = self.label_dict[anchor_label][positive_idx]
        
        negative_label = np.random.choice(self.label_count)
        while negative_label == anchor_label:
            negative_label = np.random.choice(self.label_count)

        negative_idx = np.random.choice(len(self.label_dict[negative_label]))
        negative_filename = self.label_dict[negative_label][negative_idx]

        anchor_img = self.loader(os.path.join(self.root_folder, anchor_filename))
        positive_img = self.loader(os.path.join(self.root_folder, positive_filename))
        negative_img = self.loader(os.path.join(self.root_folder, negative_filename))

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label, negative_label

    def __len__(self):
        return len(self.fp_items)

