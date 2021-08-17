import torch
import sys
import random
import shutil
from easydict import EasyDict as edict
import yaml
import numpy as np
import scipy.io as sio
import time
import os
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
sys.path.append("..")
sys.path.append("/home/hanhu/code/TIFS19/Adversarial_DA_PAD/tools")
import params

def save_checkpoint(state, is_best=False, filename='checkpoint', output='./'):
    if not os.path.exists(output):
        os.makedirs(output)
    save_file = os.path.join(output, str(filename) + '_latest.path.tar')
    save_best = os.path.join(output, str(filename) + '_best.path.tar')
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, save_best)

def Config(yaml_filename):
    with open(yaml_filename, 'r') as f:
        parser = edict(yaml.load(f))
    # for key in parser:
    #     print('{}: {}'.format(key, parser[key]))
    return parser


def drawROC(features_subject, features_cnt, epoch, output):
    [subject, samples_per, feat_dim] = features_subject.shape
    positive_total = 0
    negative_total = 0
    cum = 0
    for cnt in features_cnt:
        positive_total = positive_total + cnt * (cnt - 1) / 2
        if cum != 0:
            negative_total = negative_total + cnt * cum
        cum += cnt

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('{}\tPositive Pairs: {}\tNegative Pairs: {}'.format(cur_time, positive_total, negative_total))
    positive_similarity = np.zeros(shape=(positive_total, ), dtype=np.float32)
    negative_similarity = np.zeros(shape=(negative_total, ), dtype=np.float32)
    cnt = 0
    for sub_i in range(subject):
        for sample_i in range(features_cnt[sub_i]):
            for sample_j in range(sample_i+1, features_cnt[sub_i]):
                feat_i = features_subject[sub_i, sample_i, :]
                feat_j = features_subject[sub_i, sample_j, :]
                # print feat_i.shape
                # print feat_j.shape
                # for ele in feat_i:
                #     print ele
                # print('norm of feat i: {}'.format(np.linalg.norm(feat_i, ord=2)))
                # print('norm of feat j: {}'.format(np.linalg.norm(feat_j, ord=2)))
                similarity = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i, ord=2) * np.linalg.norm(feat_j, ord=2))
                similarity = abs(similarity)
                positive_similarity[cnt] = similarity
                cnt += 1
    cnt = 0
    for sub_i in range(subject):
        for sub_j in range(sub_i+1, subject):
            for sample_i in range(features_cnt[sub_i]):
                for sample_j in range(features_cnt[sub_j]):
                    feat_i = features_subject[sub_i, sample_i, :]
                    feat_j = features_subject[sub_j, sample_j, :]
                    similarity = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i, ord=2) * np.linalg.norm(feat_j, ord=2))
                    similarity = abs(similarity)
                    negative_similarity[cnt] = similarity
                    cnt += 1

    steps = 10000
    threshold_range = range(0, steps+1)
    threshold_range = np.true_divide(threshold_range, steps)
    false_positive_rate = np.zeros(shape=(steps+1, ), dtype=np.float32)
    true_positive_rate = np.zeros(shape=(steps+1, ), dtype=np.float32)
    
    positive_similarity = np.sort(positive_similarity)
    negative_similarity = np.sort(negative_similarity)

    positive_cnt = 0
    negative_cnt = 0
    TP = positive_total
    FN = 0
    TN = 0
    FP = negative_total
    cnt = 0
    for threshold in threshold_range:
        while positive_cnt < positive_total and positive_similarity[positive_cnt] < threshold:
            TP -= 1
            FN += 1
            positive_cnt += 1
        while negative_cnt < negative_total and negative_similarity[negative_cnt] < threshold:
            FP -= 1
            TN += 1
            negative_cnt += 1
        true_positive_rate[cnt] = TP * 1.0 / (TP + FN)
        false_positive_rate[cnt] = FP * 1.0 / (FP + TN)
        cnt += 1
    mat_name = 'epoch_' + str(epoch) + '.mat'
    if not os.path.exists(output):
        os.makedirs(output)
    mat_file = os.path.join(output, mat_name)
    sio.savemat(mat_file, {'epoch': epoch, 'true_positive_rate': true_positive_rate, 'false_positive_rate': false_positive_rate})


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)



def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construct_model(model, args):

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


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))


def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)
    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """
    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l

def remove(file):
    if os.path.exists(file): os.remove(file)

def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

def np_float32_to_uint8(x, scale=255.0):
    return (x*scale).astype(np.uint8)

def np_uint8_to_float32(x, scale=255.0):
    return (x/scale).astype(np.float32)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
