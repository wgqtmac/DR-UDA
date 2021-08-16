import torch.nn as nn
import torch
import tools.losses
import tools.triplets_utils as tu
from sklearn.cluster import KMeans
from torchvision.utils import save_image
from misc.utils import get_inf_iterator, mkdir, init_random_seed

image_size = 248
m_lambda = 0.7

def fit_disc(src_model, tgt_model, disc,
             src_loader, tgt_loader,
             opt_tgt, opt_disc,
             epochs=200,
             verbose=1):
    tgt_model.train()
    disc.train()

    # setup criterion and opt
    criterion = nn.CrossEntropyLoss()

    iternum = max(len(src_loader), len(tgt_loader))

    print('iternum={}'.format(iternum))

    ####################
    # 2. train network #
    ####################

    for epoch in range(epochs):

        data_source = get_inf_iterator(src_loader)
        data_target = get_inf_iterator(tgt_loader)

        for iters in range(iternum):
            images_src, lab_src = next(data_source)
            images_tgt, lab_tgt = next(data_target)

        # zip source and target data pair

        # data_zip = enumerate(zip(src_loader, tgt_loader))
        # for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################


            images_src = images_src.cuda()
            images_tgt = images_tgt.cuda()

            # zero gradients for opt
            opt_disc.zero_grad()

            # extract and concat features
            # feat_src = src_model.get_embedding(images_src)
            # feat_tgt = tgt_model.get_embedding(images_tgt)
            feat_src = src_model.extract_features(images_src)
            feat_tgt = tgt_model.extract_features(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = disc(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long()
            label_tgt = torch.zeros(feat_tgt.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), 0).cuda()

            # compute loss for disc
            loss_disc = criterion(pred_concat, label_concat)
            loss_disc.backward()

            # optimize disc
            opt_disc.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for opt
            opt_disc.zero_grad()
            opt_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_model.extract_features(images_tgt)

            # predict on discriminator
            pred_tgt = disc(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            opt_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if verbose and ((iters + 1) % 20 == 0):
                print("Epoch [{}/{}] - step[{}/{}]"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              epochs,
                              iters+1,
                              iternum,
                              loss_disc.item(),
                              loss_tgt.item(),
                              acc.item()))


# def fit_center(src_model, tgt_model, src_loader, tgt_loader,
#                opt_tgt, epochs=30):
#     """Center loss."""
#     ####################
#     # 1. setup network #
#     ####################
#     n_classes = 2
#     # set train state for Dropout and BN layers
#     src_model.train()
#     tgt_model.train()
#
#     src_embeddings, _ = tu.extract_embeddings(src_model, src_loader)
#
#     src_kmeans = KMeans(n_clusters=n_classes)
#     src_kmeans.fit(src_embeddings)
#
#     # src_centers = torch.FloatTensor(src_kmeans.means_).cuda()
#     src_centers = torch.FloatTensor(src_kmeans.cluster_centers_).cuda()
#
#     ####################
#     # 2. train network #
#     ####################
#
#     for epoch in range(epochs):
#         for step, (images, labels) in enumerate(tgt_loader):
#             # make images and labels variable
#             images = images.cuda()
#             labels = labels.squeeze_().cuda()
#
#             # zero gradients for opt
#             opt_tgt.zero_grad()
#
#             # compute loss for critic
#             loss = losses.center_loss(tgt_model, {"X": images, "y": labels}, src_model,
#                                       src_centers, None, src_kmeans,
#                                       None)
#             # optimize source classifier
#             loss.backward()
#             opt_tgt.step()

def fit_rec(src_model, tgt_model, src_loader, tgt_loader,
               opt_rec, current_epoch, epochs=1):
    """Reconstruction loss."""
    loss_rec = nn.MSELoss()
    loss_rec = loss_rec.cuda()
    for epoch in range(epochs):
        # zip source and target data pair

        for step, (images, labels) in enumerate(tgt_loader):
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            # image1_src = image1_src.cuda()
            # image2_src = image1_src.cuda()
            # image2_tgt = image1_src.cuda()
            # image2_tgt = image1_src.cuda()
            # images_src = torch.cat((image1_src, image2_src), 1)
            # images_tgt = torch.cat((image1_tgt, image2_tgt), 1)
            images = images.cuda()
            labels = labels.squeeze_().cuda()

            # zero gradients for opt
            opt_rec.zero_grad()

            # extract and concat features

            feat_tgt, rec_images = tgt_model(images)

            # mkdir('./recovery_image')
            save_image(images.data, './recovery_image/real' + str(current_epoch) + '.png', nrow=8)
            save_image(rec_images.data, './recovery_image/rec' + str(current_epoch) + '.png', nrow=8)

            rec_img_vec = rec_images.view(-1, 1 * image_size * image_size)
            images_vec = images.contiguous().view(-1, 1 * image_size * image_size)
            err_rec = (1 - m_lambda) * loss_rec(rec_img_vec, images_vec)
            err_rec.backward()
            opt_rec.step()

            if ((step + 1) % 20 == 0):
                print("Epoch [{}/{}] - step[{}]"
                      " rec_loss={:.5f}"
                      .format(epoch + 1,
                              epochs,
                              step+1,
                              err_rec.cpu().data.numpy()
                             ))
