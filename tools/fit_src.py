import torch.nn as nn
import losses
from losses import CenterLoss
import torch

def fit_src(model, data_loader, opt):
    loss_sum = 0.
    centerloss = CenterLoss(num_classes=2, feat_dim=2, use_gpu=True)
    for step, (image1, labels) in enumerate(data_loader):
        # make images and labels variable
        image1 = image1.cuda()
        # image2 = image2.cuda()
        labels = labels.squeeze_().cuda()

        # image_concat = torch.cat((image1, image2), 1)

        # zero gradients for opt
        opt.zero_grad()

        # feat, preds = model(image1)
        # center_loss = centerloss(feat, labels)

        # compute loss for critic
        loss = losses.triplet_loss(model, {"X": image1, "y": labels})

        # loss = loss + center_loss

        loss_sum += loss.item()

        # optimize source classifier
        loss.backward()
        opt.step()

    return {"loss": loss_sum / step}

