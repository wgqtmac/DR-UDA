

root_folder = "/home/gqwang/Spoof_Croped"
src_train_list = "4casia_adapt_list.txt"
src_val_list = "4casia_adapt_list.txt"
src_test_list = "4casia_adapt_list.txt"
src_adapt_list = "4casia_adapt_list.txt"
tgt_train_list = "2idiap_adapt_list.txt"
tgt_test_list = "2idiap_adapt_list.txt"
tgt_adapt_list = "2idiap_adapt_list.txt"
#pretrained_model = "/home/gqwang/code/Spoof_MADDA/caffe2pytorch/ResNet80_pytorch_model_lastest.pth.tar"
pretrained_model = "/home/gqwang/code/TIFS/Adversarial_DA_PAD/resnet18.pth"
cfg = "/home/gqwang/code/TIFS/Adversarial_DA_PAD/config/config.yml"
num_epochs = 20
output = "/home/gqwang/code/TIFS/Adversarial_DA_PAD/Outputs/spoof_train"
workers = 6
weight_decay = 0.0005
momentum = 0.9
display = 50

max_iter = 2000

batch_size = 64
test_batch_size = 32
test_interval = 200
samples_per = 50
train_subs = 3229
test_subs = 807
topk = 3

base_lr = 0.0001
start_epoch = 0
start_iters = 0
best_model= 12345678.9
#-------------lr_policy--------------------#
# step
lr_policy = 'step'
#policy_parameter:
gamma = 0.333
step_size = 800

d_learning_rate = 1e-3
c_learning_rate = 1e-5
beta1 = 0.5
beta2 = 0.9
adapt_num_epochs = 20
save_step_pre = 5
log_step = 50
save_step = 1
model_root = "snapshots"

d_input_dims = 256
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

manual_seed = None
save_step_pre = 1

model = "resnet18"
image_size = 256



