import torch
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import combinations


def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)
    embeddings = np.zeros((n_samples, 128))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            # image1 = image1.cuda()
            # image2 = image2.cuda()
            #
            # images = torch.cat((image1, image2), 1)
            images = images.cuda()
            target = torch.squeeze(target) #only for main1
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels


def get_triplets(embeddings, y):
    margin = 1
    D = pdist(embeddings)
    D = D.cpu()

    y = y.cpu().data.numpy().ravel()
    trip = []

    for label in set(y):
        label_mask = (y == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        neg_ind = np.where(np.logical_not(label_mask))[0]

        ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
        ap = np.array(ap)

        ap_D = D[ap[:, 0], ap[:, 1]]

        # # GET HARD NEGATIVE
        # if np.random.rand() < 0.5:
        #   trip += get_neg_hard(neg_ind, hardest_negative,
        #                D, ap, ap_D, margin)
        # else:
        trip += get_neg_hard(neg_ind, random_neg,
                             D, ap, ap_D, margin)

    if len(trip) == 0:
        ap = ap[0]
        trip.append([ap[0], ap[1], neg_ind[0]])

    trip = np.array(trip)

    return torch.LongTensor(trip)


def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors))
    D += vectors.pow(2).sum(dim=1).view(1, -1)
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D


def get_neg_hard(neg_ind,
                 select_func,
                 D, ap, ap_D, margin):
    trip = []

    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di -
                       D[torch.LongTensor(np.array([ap_i[0]])),
                         torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip


def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def semihard_negative(loss_values, margin=1):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None