import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.backends.cudnn as cudnn
from opacus.utils.module_modification import convert_batchnorm_modules

import config as c
import model
import utils as ut


# Set random seed for reproducibility
ut.set_all_seeds_as(c.seed)

# Needed for reproducibility
cudnn.deterministic = True

# increase the speed of training if you are not varying the size of image after each epoch
cudnn.benchmark = False

# to fix error with matplotlib
plt.switch_backend('agg')

# to ensure it doesn't run partly on another gpu
torch.cuda.set_device(c.cuda_n[0])
torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Device selection
device = torch.device("cuda:" + str(c.cuda_n[0]) if (torch.cuda.is_available()
                                                     and c.ngpu > 0)
                      else "cpu")

# ####Create generator object##### #
netG = model.Generator().to(device)

# DPGAN specific
if c.diff_priv:
    netG = convert_batchnorm_modules(netG)


# Print the model
print(netG)

if (device.type == 'cuda') and (c.ngpu > 1):
    netG = nn.DataParallel(netG, c.cuda_n)


saved_params_dict = torch.load(c.load_model_path, map_location=lambda storage,
                               loc: storage)

netG.load_state_dict(saved_params_dict['Generator_state_dict'])


# number of noise images to generate
test_noise = torch.randn(c.n_test_samples, c.nz, 1, 1, 1)

dataloader = data_utils.DataLoader(test_noise, batch_size=c.test_batch_size,
                                   shuffle=False)
iter = 1

# check if the folder for saving generated images exists; if not create one
if not os.path.exists(os.path.join(c.gen_path, 'patches')):
    os.makedirs(os.path.join(c.gen_path, 'patches'))
    if c.nc == 2:
        os.makedirs(os.path.join(c.gen_path, 'seg_labels'))

for i, data in enumerate(dataloader):
    noise = data.to(device)
    with torch.no_grad():
        test_fake = netG(noise).detach().cpu()

        for idx, fake in enumerate(test_fake):
            # hard thresholding for visualisation
            sample = fake.clone()
            if c.nc == 2:
                patch = sample[0]
                label = sample[1]
                label[label > c.gen_threshold] = 1
                label[label <= c.gen_threshold] = 0
                patch = ut.rescale_unet(patch)  # rescaling back to 0-255

            if c.save_nifti:
                ut.convert_and_save_to_nifti(patch.numpy(), c.gen_images_path
                                             + "fixed_fake_trial_%d_epoch_"\
                                                 "%d_sample_%d_patch.nii.gz"
                                             % (c.model_trial, c.model_epoch,
                                                iter))
            else:
                ut.save_to_npy_gz(patch.numpy(), c.gen_path + "patches/"
                                  + "fixed_fake_trial_%d_epoch_" \
                                      "%d_sample_%d_patch.npy.gz"
                                  % (c.model_trial, c.model_epoch, iter))
            if c.nc == 2:
                if c.save_nifti:
                    ut.convert_and_save_to_nifti(label.numpy(),
                                                 c.gen_path
                                                 + "fixed_fake_trial_%d_epoch_"
                                                   "%d_sample_%d_label.nii.gz"
                                                 % (c.model_trial,
                                                    c.model_epoch, iter))
                else:
                    ut.save_to_npy_gz(label.numpy(),
                                      c.gen_path + "seg_labels/"
                                      + "fixed_fake_trial_%d_epoch_"
                                        "%d_sample_%d_label.npy.gz"
                                      % (c.model_trial, c.model_epoch, iter))
            iter += 1

            if iter % 1000 == 0:
                print(f"{iter} samples have been generated!")

print(f"All samples generated and saved in folder: {c.gen_path}")
