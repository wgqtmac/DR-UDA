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

from torch.autograd import Variable
import tools.fit_src as fs
import tools.fit_tgt as fg

from lib.utils.utils import save_checkpoint
from lib.utils.utils import save_model

import params as params

import test_src as ts





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


def train_src(encoder, train_loader, val_loader):
    global lr
    global best_prec1

    lr = params.base_lr

    # encoder = construct_premodel(encoder, params)

    # encoder = torch.nn.DataParallel(encoder)
    # classifier = torch.nn.DataParallel(encoder)

    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     encoder = encoder.cuda()
    #     classifier = classifier.cuda()

    encoder.train()
    # classifier.train()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))


    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        adjust_learning_rate(optimizer, epoch, params.base_lr)

        train_dict = fs.fit_src(encoder, train_loader, optimizer)

        loss = train_dict["loss"]
        print("Epoch [{}/{}] - loss={:.5f}".format(
            epoch, params.num_epochs, loss))

        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            src_acc, HTER = ts.validate(encoder, encoder, train_loader, val_loader)

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    src_acc, HTER = ts.validate(encoder, encoder, train_loader, val_loader)

    print("{} TEST Accuracy = {:2%} HTER={:2%}\n".format("src_val_loader",
                                              src_acc,HTER))


    return encoder



def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, tgt_test_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()



    #src_encoder = torch.nn.DataParallel(src_encoder)
    #tgt_encoder = torch.nn.DataParallel(tgt_encoder)
    #critic = torch.nn.DataParallel(critic)

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = torch.optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_rec = torch.optim.Adam(tgt_encoder.parameters(),
                                     lr=params.c_learning_rate,
                                     betas=(params.beta1, params.beta2))
    optimizer_critic = torch.optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))




    ####################
    # 2. train network #
    ####################

    for epoch in range(params.adapt_num_epochs):

        """"GAN Loss."""
        fg.fit_disc(src_encoder, tgt_encoder, critic,
                    src_data_loader, tgt_data_loader,
                    opt_tgt=optimizer_tgt,
                    opt_disc=optimizer_critic,
                    epochs=1, verbose=1)


        acc_tgt = ts.validate(src_encoder, tgt_encoder,
                              src_data_loader,
                              tgt_test_loader)


        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

        """"Center Loss."""
        # fg.fit_center(src_encoder, tgt_encoder,
        #               src_data_loader, tgt_data_loader,
        #               optimizer_tgt, epochs=1)

        """"Reconstruction Loss."""
        # fg.fit_rec(src_encoder, tgt_encoder,
        #               src_data_loader, tgt_data_loader,
        #               optimizer_rec,  epoch, epochs=1)

        # acc_tgt = ts.validate(src_encoder, tgt_encoder,
        #                       src_data_loader,
        #                       tgt_test_loader)

    return tgt_encoder



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
