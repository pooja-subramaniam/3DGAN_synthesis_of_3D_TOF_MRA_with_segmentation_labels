##################################
# ####### Train settings ####### #
##################################
list_config_names = ['Trial', 'seed', 'ngpu', 'image_size', 'nz',
                     'batch_size', 'lrd', 'lrg', 'spectral_norm_D',
                     'spectral_norm_G', 'gp', 'nc', 'n_disc',
                     'beta1', 'beta2', 'num_workers', 'k_size', 'ngf', 'ndf']

trial_num = 32
continue_train = False  # if true, also update the continue train parameters
seed = 990

# Number of training epochs
num_epochs = 100

# Save model or not, how often to save
# and format to save samples
is_model_saved = True
save_n_epochs = 1
save_nifti = True  # If false saved as npy.gz

use_mixed_precision = True

# Number of workers for dataloader
workers = 6

# Spatial size of training images. All images need to be of the same size
image_size = (128, 128, 64)

################################
# ######## GPU settings ###### #
################################
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# gpu number
cuda_n = [1]

# Root directory for dataset
dataroot = "/data-raid5/pooja/HPC/Data/PEGASUS_denoised/" \
                    "50_ppp_128x128x64/"

save_model = "/data-raid5/pooja/3DGAN/Models/Trial_" + str(trial_num) + "/"
save_results = "/data-raid5/pooja/3DGAN/Results/Trial_" + str(trial_num) \
               + "/"
save_config = "/data-raid5/pooja/3DGAN/WGAN_GP_trials.csv"

####################################
# ###### parameter settings ###### #
####################################
spectral_norm_D = True  # False for GP model and true for all others
n_disc = 1
spectral_norm_G = False
gp = True
# WGAN GP hyperparamter settings
lambdaa = 10

# Batch size during training
batch_size = 4

# kernel sizes
kd = 3
kg = 3

# Number of channels in the training images. 2 here as we are also
# using the labels through the second channel
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator ngf (max)
ngf = 1024  # 1024 for c-SN-MP model and 512 for rest

# Size of feature maps in critic ndf (max)
ndf = 1024  # 1024 for c-SN-MP model and 512 for rest

# Learning rate for optimizers
lrd = 0.0004
lrg = 0.0002

# Beta hyperparameters for Adam optimizers
beta1d = 0
beta2d = 0.9
beta1g = 0
beta2g = 0.9

list_config = [trial_num, seed, ngpu, image_size, nz,
               batch_size, lrd, lrg, spectral_norm_D,
               spectral_norm_G, gp, nc,
               n_disc, beta1d, beta2d, workers, kd, ngf, ndf]

# ######################################################## #
# ######### Config params for continue training ########## #
# ######################################################## #

epoch_num_to_continue = 1
trial_num_to_continue = 32

saved_model_path = "/data-raid5/pooja/3DGAN/Models/Trial_" + \
                    str(trial_num_to_continue) + "/" \
                    + "epoch_" + str(epoch_num_to_continue) + ".pth"

# ######################################################## #
# ######### Config params for generate patches ########### #
# ######################################################## #
model_trial = 32
model_epoch = 99
test_batch_size = 8
gen_batch = 1
n_test_samples = 11750
gen_threshold = 0.4


load_model_path = "/data-raid5/pooja/3DGAN/Models/Trial_" + str(model_trial) \
                  + "/" + "epoch_" + str(model_epoch) + ".pth"
gen_path = "/data-raid5/pooja/3DGAN/generated/Trial_" + str(model_trial) \
           + "/" + "gen_images_epoch_" + str(model_epoch) \
            + "_threshold_" + str(gen_threshold) \
            + "_" + str(gen_batch) + "/train/for_segmentation/train/"
# + "/train/" #


# Root directory for dataset
dataroot = "/data-raid5/pooja/HPC/Data/PEGASUS_denoised/" \
                    "50_ppp_128x128x64/"

save_model = "/data-raid5/pooja/3DGAN/Models/Trial_" + str(trial_num) + "/"
save_results = "/data-raid5/pooja/3DGAN/Results/Trial_" + str(trial_num) \
               + "/"
save_config = "/data-raid5/pooja/3DGAN/WGAN_GP_trials.csv"

####################################
# ###### parameter settings ###### #
####################################
spectral_norm_D = True
n_disc = 1
spectral_norm_G = False
gp = True
# WGAN GP hyperparamter settings
lambdaa = 10

# Batch size during training
batch_size = 4

# kernel sizes
kd = 3
kg = 3

# Number of channels in the training images. 2 here as we are also
# using the labels through the second channel
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator ngf * 8 (max)
ngf = 1024

# Size of feature maps in discriminator ndf * 8 (max)
ndf = 1024

# Learning rate for optimizers
lrd = 0.0004
lrg = 0.0002

# Beta hyperparameters for Adam optimizers
beta1d = 0
beta2d = 0.9
beta1g = 0
beta2g = 0.9

list_config = [trial_num, seed, ngpu, image_size, nz,
               batch_size, lrd, lrg, spectral_norm_D,
               spectral_norm_G, gp, nc,
               n_disc, beta1d, beta2d, workers, kd, ngf, ndf]

# ######################################################## #
# ######### Config params for continue training ########## #
# ######################################################## #

epoch_num_to_continue = 1
trial_num_to_continue = 32

saved_model_path = "/data-raid5/pooja/3DGAN/Models/Trial_" + \
                    str(trial_num_to_continue) + "/" \
                    + "epoch_" + str(epoch_num_to_continue) + ".pth"

# ######################################################## #
# ######### Config params for generate patches ########### #
# ######################################################## #
model_trial = 32
model_epoch = 99
test_batch_size = 8
gen_batch = 1
n_test_samples = 11750
gen_threshold = 0.3


load_model_path = "/data-raid5/pooja/3DGAN/Models/Trial_" + str(model_trial) \
                  + "/" + "epoch_" + str(model_epoch) + ".pth"
gen_path = "/data-raid5/pooja/3DGAN/generated/Trial_" + str(model_trial) \
           + "/" + "gen_images_epoch_" + str(model_epoch) \
            + "_threshold_" + str(gen_threshold) \
            + "_" + str(gen_batch) + "/train/for_segmentation/train/"
# + "/train/" #
