import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    
    def forward(self, anchor, positive, negative, size_average=True):
        dist_ap = (anchor - positive).pow(2).sum(1)
        dist_an = (anchor - negative).pow(2).sum(1)
        losses = F.relu(dist_ap - dist_an + self.margin)
        return losses.mean() if size_average else losses.sum()


class PairTripletLoss(nn.Module):
    """
        Ohem Triplet loss
        Takes two samples of every subject with online hard examples mining
    """

    def __init__(self, margin):
        super(PairTripletLoss, self).__init__()
        self.margin = margin


    def forward(self, pair1, pair2, size_average=True):
        nums, dim = pair1.shape
        avg_loss = 0.
        all_loss = 0.
        losses = 0. 
        cnt = 0
        for i in range(nums):
            dist_ap = (pair1[i] - pair2[i]).pow(2).sum()
            sub_loss_np = 0.
            sub_loss = 0.
            for j in range(nums):
                if i == j:
                    continue
                dist_an1 = (pair1[i] - pair1[j]).pow(2).sum()
                loss1 = F.relu(dist_ap - dist_an1 + self.margin)
                if loss1.data.cpu().numpy()[0] >= sub_loss_np:
                    sub_loss_np = loss1.data.cpu().numpy()[0]
                    sub_loss = loss1
                all_loss += loss1
                cnt += 1

                dist_an2 = (pair1[i] - pair2[j]).pow(2).sum()
                loss2 = F.relu(dist_ap - dist_an2 + self.margin)
                if loss2.data.cpu().numpy()[0] >= sub_loss_np:
                    sub_loss_np = loss2.data.cpu().numpy()[0]
                    sub_loss = loss2
                all_loss += loss2
                cnt += 1
            losses += sub_loss


        sub_loss = losses / nums
        all_loss = all_loss / cnt

        return sub_loss, all_loss


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss