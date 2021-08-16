"""Main script for ADDA."""
import os
import params as params
from lib.model.discriminator import Discriminator
from lib.model.ResNet18 import resnet18
from lib.model.CBAM_resnet import resnet18_cbam
from lib.model.Reconstruction_model import VAE, Encoder, Decoder
from lib.utils.utils import init_model, init_random_seed
import torch
import torch._utils
import torchvision.transforms as transforms
from datasets.ImgLoader import ImgLoader
from train_val import eval_src, eval_tgt

from train import train_src
from train import train_tgt
# from adapt import train_tgt
import test_src as ts

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)




    # load dataset
    src_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_train_list),
                              transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.RandomCrop(248),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(15),
                                  transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                  transforms.ToTensor()
                              ]))
    weights = [3 if label == 1 else 1 for data, label in src_dataset.items]
    from torch.utils.data.sampler import WeightedRandomSampler

    sampler = WeightedRandomSampler(weights,
                                    num_samples=len(src_dataset.items),
                                    replacement=True)
    src_loader = torch.utils.data.DataLoader(src_dataset,
                                               batch_size=params.batch_size,
                                               num_workers=2,
                                               sampler=sampler,
                                               drop_last=True,
                                               pin_memory=True)

    src_val_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_val_list),
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(248),
                                transforms.ToTensor()
                            ]), stage='Test')
    src_val_loader = torch.utils.data.DataLoader(src_val_dataset,
                                             batch_size=params.test_batch_size,
                                             num_workers=2,
                                             pin_memory=True)

    src_test_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_test_list),
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(248),
                                    transforms.ToTensor()
                                ]), stage='Test')
    src_test_loader = torch.utils.data.DataLoader(src_test_dataset,
                                                 batch_size=params.test_batch_size,
                                                 num_workers=2,
                                                 pin_memory=True)

    src_adapt_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_adapt_list),
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(248),
                                     transforms.ToTensor()
                                 ]), stage='Test')
    src_adapt_loader = torch.utils.data.DataLoader(src_test_dataset,
                                                  batch_size=params.batch_size,
                                                  num_workers=2,
                                                  shuffle=True,
                                                  pin_memory=True)

    # tgt_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.tgt_train_list),
    #                         transforms.Compose([
    #                             transforms.Resize(256),
    #                             transforms.RandomCrop(248),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor()
    #                         ]))

    tgt_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.tgt_train_list),
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(248),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                transforms.ToTensor()
                            ]))

    tgt_loader = torch.utils.data.DataLoader(tgt_dataset,
                                             batch_size=params.batch_size,
                                             num_workers=2,
                                             shuffle=True,
                                             pin_memory=True)

    tgt_var_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.tgt_test_list),
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(248),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ]))
    tgt_var_loader = torch.utils.data.DataLoader(tgt_dataset,
                                             batch_size=params.batch_size,
                                             num_workers=2,
                                             shuffle=True,
                                             pin_memory=True)

    tgt_adapt_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.tgt_adapt_list),
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(248),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                ]))
    tgt_adapt_loader = torch.utils.data.DataLoader(tgt_dataset,
                                                 batch_size=params.batch_size,
                                                 num_workers=2,
                                                 shuffle=True,
                                                 pin_memory=True)

    # load dataset
    # src_data_loader = get_data_loader(params.src_dataset)
    # src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    # tgt_data_loader = get_data_loader(params.tgt_dataset)
    # tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    # src_encoder = construct_model(params)

    src_encoder = init_model(net=resnet18(),
                             restore=params.src_encoder_restore)

    # src_classifier = init_model(net=ResNetClassifier(),
    #                            restore=params.src_classifier_restore)

    tgt_encoder = init_model(net=resnet18(),
                             restore=params.tgt_encoder_restore)

    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    # print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)


    if not (src_encoder.restored  and
            params.src_model_trained):
        src_encoder = train_src(
            src_encoder, src_loader, src_test_loader)

    # src_acc, src_HTER = ts.validate(src_encoder, src_encoder, src_loader, src_test_loader)

    # print(">>> source only <<<")
    # print("{} TEST Accuracy = {:2%} HTER = {:2%}\n".format("src_val_loader",
    #                                           src_acc, src_HTER))
    #
    # tgt_acc, tgt_HTER = ts.validate(src_encoder, src_encoder, src_loader, tgt_var_loader)

    # print("{} TEST Accuracy = {:2%} HTER = {:2%}\n".format("tgt_val_loader",
    #                                           tgt_acc, tgt_HTER))

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_adapt_loader, tgt_loader, tgt_var_loader)
    # # eval source model
    # print("=== Evaluating classifier for source domain ===")
    # eval_tgt(src_encoder, src_classifier, src_test_loader)
    #
    # print(">>> source only <<<")
    # eval_tgt(src_encoder, src_classifier, tgt_adapt_loader)
    #
    # # train target encoder by GAN
    # # print("=== Training encoder for target domain ===")
    # # print(">>> Target Encoder <<<")
    # # print(tgt_encoder)
    # # print(">>> Critic <<<")
    # # print(critic)
    #
    # # init weights of target encoder with those of source encoder
    # if not tgt_encoder.restored:
    #     tgt_encoder.load_state_dict(src_encoder.state_dict())
    #
    # if not (tgt_encoder.restored and critic.restored and
    #         params.tgt_model_trained):
    #     tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
    #                             src_adapt_loader, tgt_loader, src_classifier)
    #
    #
    #
    # # eval target encoder on test set of target dataset
    # print("=== Evaluating classifier for encoded source domain ===")
    # eval_tgt(src_encoder, src_classifier, src_test_loader)
    # print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> source only <<<")
    # eval_tgt(src_encoder, src_classifier, tgt_adapt_loader)
    # print(">>> domain adaption <<<")
    # eval_tgt(tgt_encoder, src_classifier, tgt_adapt_loader)

