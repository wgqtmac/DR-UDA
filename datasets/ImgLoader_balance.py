import os
import torch
import torch.utils.data as data
from PIL import Image
import cv2
from datasets.data_helper import *
from datasets.augmentation import *
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

def YCbCr_loader(path):
    return Image.open(path).convert('YCbCr')

class ImgLoader(data.Dataset):
    def __init__(self, root_folder, list_file, mode, transform=None, loader1=default_loader, loader2=YCbCr_loader):
        self.root_folder = root_folder
        self.mode = mode
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
        print('\nmode: ' + mode)
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



class FDDataset(data.Dataset):
    def __init__(self, root_folder, list_file, mode, image_size=256, augment=None, augmentor=None, balance=True):
        super(FDDataset, self).__init__()

        self.mode = mode
        self.augment = augment
        self.augmentor = augmentor
        self.balance = balance

        self.channels = 3
        # self.train_image_path = TRN_IMGS_DIR
        # self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size

        self.set_mode(self.mode, root_folder, list_file)

    def set_mode(self, mode, root_folder, list_file):
        self.mode = mode

        if self.mode == 'test':
            self.test_list = load_test_list(root_folder, list_file)
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        elif self.mode == 'val':
            self.val_list = load_val_list(root_folder, list_file)
            self.num_data = len(self.val_list)
            print('set dataset mode: test')

        elif self.mode == 'train':
            self.train_list = load_train_list(root_folder, list_file)

            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)

            if self.balance:
                self.train_list = transform_balance(self.train_list)
            print('set dataset mode: train')

        print(self.num_data)

    def __getitem__(self, index):

        if self.mode == 'train':
            if self.balance:
                if random.randint(0,1)==0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0,len(tmp_list)-1)
                image, label = tmp_list[pos]
            else:
                image, label = self.train_list[index]

        elif self.mode == 'val':
            image, label = self.val_list[index]

        elif self.mode == 'test':
            image, label = self.test_list[index]


        img_path = os.path.join(DATA_ROOT, image)
        image = cv2.imread(img_path,1)
        image = cv2.resize(image,(self.image_size,self.image_size))

        if self.mode == 'train':
            image = self.augment(image)

            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            image = self.augment(image)


            n = len(image)

            # image = np.concatenate(image,axis=0)
            #
            # image = np.transpose(image, (2, 0, 1))
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            # image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'test':

            image = self.augment(image)
            n = len(image)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            # image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))


    def __len__(self):
        return self.num_data



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    augment = train_augumentor
    dataset = FDDataset(mode='train', image_size=256, augment=augment)
    train_loader = torch.utils.data.DataLoader(dataset,
                              shuffle=True,
                              batch_size=64,
                              drop_last=True,
                              num_workers=8)
    for input, truth in train_loader:


        # one iteration update  -------------
        input = input.cuda()
        truth = truth.cuda()
