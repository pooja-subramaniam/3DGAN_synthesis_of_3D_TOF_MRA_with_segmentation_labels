'''
Configs for generating features
'''
import os

manual_seed = 1000

# extracted features of generated or real data
generated = True # False for real data

# if generated then trial and epoch number
gan_trial_num = 22
gan_epoch_num = 99
threshold = 0.4
batch = 1
split = 'train'
ppp = 100
image_size_str = '128x128x64'
patches_dataset = str(ppp)+'_ppp_'+image_size_str
# data paths
data_dir_real = '/data-raid5/pooja/HPC/Data/PEGASUS_denoised/' + patches_dataset + '/' + split + '/'
data_dir_gen = '/data-raid5/pooja/3DGAN/generated/Trial_' + str(gan_trial_num) + \
                '/gen_images_epoch_' + str(gan_epoch_num) + '_threshold_' + str(threshold) + '_' + str(batch) +'/train/' #

# paths for real and generated features to be saved
data_dir_real_features = '/data-raid5/pooja/3DGAN/features/real' + patches_dataset + '/'
data_dir_gen_features = '/data-raid5/pooja/3DGAN/features/generated/Trial_' + str(gan_trial_num) + \
                      '/epoch_' + str(gan_epoch_num) + '_threshold_' + str(threshold) + '/'

if generated:
    dataroot = data_dir_gen
    save_features_dir = data_dir_gen_features
else:
    dataroot = data_dir_real
    save_features_dir = data_dir_real_features

if not os.path.exists(save_features_dir):
    os.makedirs(save_features_dir)

# path to the pretrained model weights
pretrained_model_path = '/data-raid5/pooja/3DGAN/MedResnet_pretrained/resnet_10_23dataset.pth'

# Spatial size of training images. All images need to be of the same size
image_size = (128, 128, 64) 

# Number of workers for dataloader
workers = 6
num_images = 2350

################################
# ######## GPU settings ###### #
################################
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# gpu number
cuda_n = [0]
