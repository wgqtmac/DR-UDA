import os
import torch
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def YCbCr_loader(path):
    return Image.open(path).convert('YCbCr')

class ImgLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader1=default_loader, loader2=YCbCr_loader, stage='Train'):
        self.root_folder = root_folder
        self.loader1 = loader1
        self.loader2 = YCbCr_loader
        self.transform = transform

        items = []

        fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
        for file_name, label in fp_items:
            if os.path.isfile(os.path.join(root_folder, file_name)):
                tup = (file_name, int(label))
                items.append(tup)
        self.items = items
        print('\nStage: ' + stage)
        print('The number of samples: {}'.format(len(items)))

    def __getitem__(self, index):
        image, label = self.items[index]
        img1 = self.loader1(os.path.join(self.root_folder, image))
        img2 = self.loader2(os.path.join(self.root_folder, image))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, label

    def __len__(self):
        return len(self.items)
