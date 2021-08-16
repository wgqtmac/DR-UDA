import os

import torch
import torch.optim as optim
from torch import nn

import torch.nn.functional as F

import tools.triplets_utils as tu


def triplet_loss(model, batch):
    model.train()
    emb,_ = model(batch["X"].cuda())
    y = batch["y"].cuda()

    with torch.no_grad():
        triplets = tu.get_triplets(emb, y)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 1.)

    return losses.mean()


def center_loss(tgt_model, batch, src_model, src_centers, tgt_centers,
                src_kmeans, tgt_kmeans, margin=1):
    # triplets = self.triplet_selector.get_triplets(embeddings, target, embeddings_adv=embeddings_adv)
    # triplets = triplets.cuda()

    # f_N = embeddings_adv[triplets[:, 2]]

    f_N_clf = tgt_model.convnet(batch["X"].cuda()).view(batch["X"].shape[0], -1)
    f_N = tgt_model.fc(f_N_clf.detach())

    # est.predict(f_N.cpu().numpy())
    y_src = src_kmeans.predict(f_N.detach().cpu().numpy())
    # ap_distances = (emb_centers[None] - f_N[:,None]).pow(2).min(1)[0].sum(1)
    ap_distances = (src_centers[y_src] - f_N).pow(2).sum(1)
    # ap_distances = (f_C[None] - f_N[:,None]).pow(2).sum(1).sum(1)

    # an_distances = 0
    losses = ap_distances.mean()

    # y_tgt = tgt_kmeans.predict(f_N.detach().cpu().numpy())
    # ap_distances = (tgt_centers[y_tgt] - f_N).pow(2).max(1)[0]

    # losses += ap_distances.mean()*0.1

    f_P = src_model(batch["X"].cuda())
    # an_distances = (f_P - f_N).pow(2).sum(1)
    # losses -= an_distances.mean() * 0.1

    return losses


def rec_loss():

    losses = nn.MSELoss()

    return losses


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
