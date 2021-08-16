import numpy as np
import sys
import os
import argparse

import time
import shutil
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
# import torchvision.datasets.ImageFolder as ImageFolder
import torchvision.models as models
from torch.autograd import Variable

import tools._init_paths

from datasets.TripletDataLoader import TripletDataLoader
from datasets.ImgLoader import ImgLoader
from datasets.PairDataLoader import PairDataLoader
from model.ResNet80 import resnet80
from utils.utils import Config
from utils.utils import drawROC
from utils.utils import save_checkpoint
from utils.utils import save_model

import params



def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a ResNet80 network.')

    parser.add_argument('--root_folder', dest='root_folder', required=True, help='root folder to load data')
    parser.add_argument('--train_list', dest='train_list', required=True, help='file list for train')
    parser.add_argument('--test_list', dest='test_list', required=True, help='file list for test')
    parser.add_argument('--cfg', dest='cfg', required=True, help='Config file for training(and optionally testing)')
    parser.add_argument('--pretrained_model', dest='pretrained_model', required=True, help='the pretrained mdoel')
    parser.add_argument('--batch_size', default=250, type=int, help='batch size of data')

    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch count. Epoch is 0-indexed')
    parser.add_argument('--num_epochs', dest='num_epochs', default=50, type=int, help='Number of epochs to train.')
    parser.add_argument('--output', dest='output', required=True, help='trained model to save.')

    return parser.parse_args()




def construct_model(args):
    model = resnet80(num_classes=2)
    # model = torch.nn.DataParallel(model)

    pretrained_model = torch.load(args.pretrained_model)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model_keys = model.state_dict().keys()
    for k, v in pretrained_model.items():
        if 'fc2' in k:
            continue
        if not k in model_keys:
            continue
        new_state_dict[k] = v
    state_dict = model.state_dict()
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)

    return model


def construct_premodel(model, args):
    # model = resnet80(num_classes=2)
    # model = torch.nn.DataParallel(model)

    pretrained_model = torch.load(args.pretrained_model)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model_keys = model.state_dict().keys()
    for k, v in pretrained_model.items():
        if 'fc2' in k:
            continue
        if not k in model_keys:
            continue
        new_state_dict[k] = v
    state_dict = model.state_dict()
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)

    return model

# torch.cuda.set_device(3)

lr = 0.
best_prec1 = 0.

# def main():
#
#     #args = parse_args()
#     #cfg = Config(args.cfg)
#     # for key in vars(args):
#     #     print('{}: {}'.format(key, getattr(args, key)))
#     # for key in cfg:
#     #     print('{}: {}'.format(key, cfg[key]))
#
#     global lr
#     global best_prec1
#     #lr = cfg.base_lr
#     lr = params.base_lr
#
#     train_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.train_list),
#                               transforms.Compose([
#                                   transforms.Resize(256),
#                                   transforms.RandomCrop(248),
#                                   transforms.RandomHorizontalFlip(),
#                                   transforms.ToTensor()
#                               ]))
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                             batch_size=params.batch_size,
#                                             num_workers=2,
#                                             shuffle=True,
#                                             pin_memory=True)
#
#     val_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.test_list),
#                             transforms.Compose([
#                                 transforms.Resize(256),
#                                 transforms.CenterCrop(248),
#                                 transforms.ToTensor()
#                             ]), stage='Test')
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                             batch_size=params.test_batch_size,
#                                             num_workers=2,
#                                             pin_memory=True)
#
#     model = construct_model(params)
#     model = torch.nn.DataParallel(model)
#
#     use_gpu = torch.cuda.is_available()
#     if use_gpu:
#         model = model.cuda()
#
#     criterion = nn.CrossEntropyLoss().cuda()
#     # optimizer = torch.optim.SGD(model.parameters(), cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
#     optimizer = torch.optim.Adam(model.parameters(), lr=params.base_lr, betas=(0.9, 0.99))
#
#     for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
#         adjust_learning_rate(optimizer, epoch, params.base_lr)
#
#         # train for one epoch
#         train(train_loader, model, criterion, optimizer, epoch)
#
#         # evaludate on validation set
#         prec1 = validate(val_loader, model, criterion, epoch)
#
#         # remeber best prec@1 and save checkpoint
#         is_best = prec1 > best_prec1
#         best_prec1 = max(prec1, best_prec1)
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'best_prec1': best_prec1,
#             'optimizer': optimizer.state_dict()
#         }, is_best, epoch + 1, params.output)


def train_src(encoder, classifier, train_loader, val_loader):


    global lr
    global best_prec1

    lr = params.base_lr

    encoder = construct_premodel(encoder, params)

    # encoder = torch.nn.DataParallel(encoder)
    # classifier = torch.nn.DataParallel(encoder)

    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     encoder = encoder.cuda()
    #     classifier = classifier.cuda()


    encoder.train()
    classifier.train()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss().cuda()


    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        adjust_learning_rate(optimizer, epoch, params.base_lr)

        # train for one epoch
        train_batch(train_loader, encoder, classifier, criterion, optimizer, epoch)

        # evaludate on validation set
        prec1 = validate_batch(val_loader, encoder, classifier, criterion, epoch)

        # remeber best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

        save_checkpoint({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, epoch + 1, params.output)

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.
    acc = 0.

    # set loss function
    criterion = nn.CrossEntropyLoss()

    #encoder = torch.nn.DataParallel(encoder)
    #classifier = torch.nn.DataParallel(encoder)

    # evaluate network
    for (images, labels) in data_loader:
        labels = labels.cuda(async=True)
        images = images.cuda()
        images_var, label_var = Variable(images), Variable(labels)

        preds = classifier(encoder(images_var))

        #loss += criterion(preds, labels).data[0]
        loss += criterion(preds[1], label_var).item()

        pred_cls = preds[1].data.max(1)[1]
        acc += pred_cls.eq(label_var.data).cpu().sum()

    loss /= len(data_loader)
    acc = float(acc) / float(len(data_loader.dataset))



    print("Avg Loss = {}, Avg Accuracy = {:2%} ".format(loss, acc))

def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    # evaluate network
    for (images, labels) in data_loader:
        labels = labels.cuda(async=True)
        images = images.cuda()
        images_var, label_var = Variable(images), Variable(labels)

        preds = classifier(encoder(images_var))
        loss += criterion(preds[1], label_var).item()

        pred_cls = preds[1].data.max(1)[1]
        acc += pred_cls.eq(label_var.data).cpu().sum()

        target_list = labels.cpu().numpy()
        pred_list = pred_cls.cpu().numpy()


        for i in range(len(target_list)):
            if target_list[i] == 1 and pred_list[i] == 1:
                TP += 1
            elif target_list[i] == 0 and pred_list[i] == 0:
                TN += 1
            elif target_list[i] == 1 and pred_list[i] == 0:
                FN += 1
            elif target_list[i] == 0 and pred_list[i] == 1:
                FP += 1

    loss /= len(data_loader)
    acc = float(acc) / float(len(data_loader.dataset))

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print('TP:{}, TP+FN:{}, TN:{}, TN+FP:{}'.format(TP, TP + FN, TN, TN + FP))

    TP_rate = float(TP / (TP + FN))
    TN_rate = float(TN / (TN + FP))

    HTER = 1 - (TP_rate + TN_rate) / 2

    print('TP rate:{}, TN rate:{}, HTER:{}'.format(float(TP / (TP + FN)), float(TN / (TN + FP)), HTER))



def train_batch(train_loader, encoder, classifier, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time =AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()



    loss = 0.
    cnt = 0

    end = time.time()
    for batch_idx, (input_data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prepare the data
        target = target.cuda(async=True)
        input_data = input_data.cuda()
        input_var, target_var = Variable(input_data), Variable(target)

        output = classifier(encoder(input_var))

        loss = criterion(output[1], target_var)

        prec1 = accuracy(output[1].data, target, topk=(1,))
        losses.update(loss.item(), input_data.size(0))

        #top1.update(prec1[0].cpu().numpy()[0], input_data.size(0))
        top1.update(prec1[0].cpu().numpy(), input_data.size(0))


        # compute gradient and do backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'lr {lr}'.format(
                  epoch, batch_idx + 1, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, top1=top1, lr=lr))


    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('\n{}\tEpoch: {}\tLoss: {}\t Prec@1 {}\n'.format(cur_time, epoch, losses.avg, top1.avg))


def validate_batch(val_loader, encoder, classifier, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    encoder.eval()
    classifier.eval()

    end = time.time()
    for batch_idx, (input_data, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_data = input_data.cuda()
        # input_var = Variable(input_data, volatile=True)
        # target_var = Variable(target, volatile=True)

        input_var = Variable(input_data, requires_grad=False)
        target_var = Variable(target, requires_grad=False)

        output = classifier(encoder(input_var))
        loss = criterion(output[1], target_var)

        prec1 = accuracy(output[1].data, target, topk=(1,))
        losses.update(loss.item(), input_data.size(0))
        top1.update(prec1[0].cpu().numpy(), input_data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} (batch_time.avg:.3f)\t'
                  'Loss {loss.val:.4f} (loss.avg:.4f)\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                batch_idx + 1, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('\n{}\tVal:\tEpoch: {}\tLoss: {}\t Prec@1 {}\n'.format(cur_time, epoch, losses.avg, top1.avg))



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time =AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    loss = 0.
    cnt = 0

    end = time.time()
    for batch_idx, (input_data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prepare the data
        target = target.cuda(async=True)
        input_var, target_var = Variable(input_data), Variable(target)

        output = model(input_var)

        loss = criterion(output[1], target_var)

        prec1 = accuracy(output[1].data, target, topk=(1,))
        losses.update(loss.item(), input_data.size(0))

        #top1.update(prec1[0].cpu().numpy()[0], input_data.size(0))
        top1.update(prec1[0].cpu().numpy(), input_data.size(0))


        # compute gradient and do backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'lr {lr}'.format(
                  epoch, batch_idx + 1, len(train_loader), batch_time=batch_time, 
                  data_time=data_time, loss=losses, top1=top1, lr=lr))


    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    print('\n{}\tEpoch: {}\tLoss: {}\t Prec@1 {}\n'.format(cur_time, epoch, losses.avg, top1.avg))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    
    end = time.time()
    for batch_idx, (input_data, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        #input_var = Variable(input_data, volatile=True)
        #target_var = Variable(target, volatile=True)
        
        input_var = Variable(input_data, requires_grad=False)
        target_var = Variable(target, requires_grad=False)

        output = model(input_var)
        loss = criterion(output[1], target_var)

        prec1 = accuracy(output[1].data, target, topk=(1,))
        losses.update(loss.item(), input_data.size(0))
        top1.update(prec1[0].cpu().numpy(), input_data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} (batch_time.avg:.3f)\t'
                  'Loss {loss.val:.4f} (loss.avg:.4f)\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  batch_idx + 1, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    print('\n{}\tVal:\tEpoch: {}\tLoss: {}\t Prec@1 {}\n'.format(cur_time, epoch, losses.avg, top1.avg))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr = base_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# if __name__ == '__main__':
#     main()
    # extract_feature('50checkpoint.pth.tar')
