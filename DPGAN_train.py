import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os
import gc

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pytorch_model_summary import summary


import torchcsprng as prng
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


import config as c
import model
import utils as ut
from dataset import GANDataset


# Set random seed for reproducibility
ut.set_all_seeds_as(c.seed)

# Needed for reproducibility
cudnn.deterministic = True

# increase the speed of training if you are not
# varying the size of image after each epoch
cudnn.benchmark = True

# to fix error with matplotlib
plt.switch_backend('agg')

# to ensure it doesn't run partly on another gpu
torch.cuda.set_device(c.cuda_n[0])

# workaround for a unhelpful cudann error
torch.set_default_tensor_type("torch.cuda.FloatTensor")

# get patches and labels paths
path_patches = glob.glob(c.dataroot+"train/patches/"+"*.gz")
path_labels = glob.glob(c.dataroot+"train/seg_labels/"+"*.gz")

dataset = GANDataset(path_patches, path_labels)


dataloader = data_utils.DataLoader(dataset,
                                   num_workers=c.workers,
                                   batch_sampler=UniformWithReplacementSampler(
                                    num_samples=len(dataset),
                                    sample_rate=c.batch_size / len(dataset))
                                   )
# Device selection
device = torch.device("cuda:" + str(c.cuda_n[0]) if (torch.cuda.is_available()
                                                     and
                                                     c.ngpu > 0) else "cpu")

# ####Create generator object##### #
netG = model.Generator().to(device)

# #### Create discriminator object #### #
netD = model.Discriminator().to(device)

# Batchnorm which holds information specific to batch 
# is not used in differential privacy to avoid information leak
netD = convert_batchnorm_modules(netD)
netG = convert_batchnorm_modules(netG)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (c.ngpu > 1):
    netG = nn.DataParallel(netG, c.cuda_n)
    netD = nn.DataParallel(netD, c.cuda_n)

# Print the models
print(netG)
print(summary(netG, torch.zeros(c.batch_size, c.nz, 1, 1, 1).to(device)))

print(netD)
print(summary(netD, torch.zeros(c.batch_size, c.nc, c.image_size[0],
                                c.image_size[1], c.image_size[2]).to(device)))

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(c.batch_size, c.nz, 1, 1, 1, device=device)

# Setup Adam optimizers for both G and D

optimizerD = optim.Adam(netD.parameters(), lr=c.lrd,
                        betas=(c.beta1d, c.beta2d))
optimizerG = optim.Adam(netG.parameters(), lr=c.lrg,
                        betas=(c.beta1g, c.beta2g))

# Attaching privacy engine to critic optimizer
privacy_engine = PrivacyEngine(netD, sample_rate=c.batch_size / c.num_images,
                               alphas=c.alphas, noise_multiplier=c.noise_m,
                               max_grad_norm=c.max_norm_dp, secure_rng=c.secure_rng,
                               target_delta=c.delta)

privacy_engine.attach(optimizerD)
torch.set_default_tensor_type("torch.FloatTensor")
epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(c.delta)
print("(epsilon = %.2f, delta = %.2f) for alpha = %.2f"
      % (epsilon, c.delta, best_alpha))
torch.set_default_tensor_type("torch.cuda.FloatTensor")


# print and save config params in trials.csv
print("Configuration for the run: \n")
print(dict(zip(c.list_config_names, c.list_config)))
ut.save_config(c.save_config, c.list_config_names, c.list_config)


# Lists to keep track of progress
G_losses = []
D_losses = []
Wasserstein_D = []
start_epoch = 0
epsilons = []

iters = 0
duration = 0

# Training Loop

print("Starting Training Loop...")

# For each epoch
for epoch in range(start_epoch, c.num_epochs):
    epoch_start_time = time.time()
    # For each batch in the dataloader
    errD_iter = []
    errG_iter = []
    Wasserstein_D_iter = []
    batch_start_time = time.time()

    for i, data in enumerate(dataloader, 0):
        # for each iteration in the epoch
        errD_disc_iter = []
        Wasserstein_D_disc_iter = []
        batch_duration = 0
        
        # Format batch of real data
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)

        # ########################## #
        # (1) Update D network n_disc times:
        # with wasserstein_loss clipping weights
        # ######################### #

        # updating the critic n_disc number of times
        # before 1 update of generator - with 3D this is usually 1
        for k in range(c.n_disc):
            netD.zero_grad()
            netG.zero_grad()

            # For training with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, c.nz, 1, 1, 1, device=device)            

            # Train with real batch
            errD_real = netD(real_cpu)
            # Calculate loss on all-real batch
            errD_real = -errD_real.view(-1).mean()
            
            errD_real.backward()
            optimizerD.step()

            # Generate fake image batch with G
            fake = netG(noise)

            # Train with fake batch
            errD_fake = netD(fake.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = errD_fake.view(-1).mean()

            errD_fake.backward()
            optimizerD.step()

            errD = errD_fake.item() + errD_real.item()

            # Wasserstein GAN Lipschitz continuity - Arjovsky et al. 2017
            if c.add_clip:
                for parameter in netD.parameters():
                    parameter.data.clamp_(-c.clip_param_W,
                                          c.clip_param_W)

            errD_disc_iter.append(errD)

        gc.collect()

        errD_disc_avg = np.mean(np.array(errD_disc_iter))

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()

        noise = torch.randn(b_size, c.nz, 1, 1, 1, device=device)  

        fake = netG(noise)

        output_fake = netD(fake)

        errG = -output_fake.view(-1).mean()

        # Calculate gradients for G
        errG.backward()
        optimizerG.step()

        # update the iteration errors
        errD_iter.append(errD_disc_avg)
        errG_iter.append(errG.item())

        iters += 1
        gc.collect()
        
        # print after every 100 batches
        if i % 100 == 0:
            print("[%d/%d] batches done!\n" % (i + 1, len(dataset) // c.batch_size))
            batch_end_time = time.time()
            batch_duration = batch_duration + batch_end_time - batch_start_time
            print("Training time for", i + 1, "batches: ", batch_duration / 60, " minutes.")

    print(" End of Epoch %d \n" % epoch)

    # Output training stats after each epoch
    avg_errD = np.mean(np.array(errD_iter))
    avg_errG = np.mean(np.array(errG_iter))

    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
          % (epoch, c.num_epochs, avg_errD, avg_errG))

    # Save Losses and outputs for plotting later
    G_losses.append(avg_errG.item())
    D_losses.append(avg_errD.item())

    if not os.path.exists(c.save_results):
        os.makedirs(c.save_results)

    np.save(c.save_results + 'G_losses.npy', np.asarray(G_losses))
    np.save(c.save_results + 'D_losses.npy', np.asarray(D_losses))

    # save epsilon values for each epoch
    torch.set_default_tensor_type("torch.FloatTensor")
    epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(c.delta)
    print(
        "(epsilon = %.2f, delta = %.2f) for alpha = %.2f"
        % (epsilon, c.delta, best_alpha)
    )
    epsilons.append(epsilon)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    np.save(c.save_results + "epsilons.npy", np.asarray(epsilons))

    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fixed_fake = netG(fixed_noise).detach().cpu()

    sample_idx = [0, 1, 2, 3]

    for idx in sample_idx:
        # hard thresholding for visualisation
        sample = fixed_fake[idx].clone()
        if c.save_nifti:
            ut.convert_and_save_to_nifti(sample[0].to(dtype=torch.float32).numpy(),
                                         c.save_results + "fixed_fake_while_training_epoch_%d_sample_%d_patch.nii.gz"
                                         % (epoch, idx))
            if c.nc == 2:
                ut.convert_and_save_to_nifti(sample[1].to(dtype=torch.float32).numpy(),
                                             c.save_results +
                                             "fixed_fake_while_training_epoch_%d_sample_%d_label.nii.gz"
                                             % (epoch, idx))

    # save model parameters'
    if c.is_model_saved:
        if not os.path.exists(c.save_model):
            os.makedirs(c.save_model)
        if (epoch+1) % c.save_n_epochs == 0:
            torch.save({'Discriminator_state_dict': netD.state_dict(),
                        'Generator_state_dict': netG.state_dict(),
                        'OptimizerD_state_dict': optimizerD.state_dict(),
                        'OptimizerG_state_dict': optimizerG.state_dict(),
                        }, c.save_model + "epoch_{}.pth".format(epoch))

    # plot and save G_loss, D_loss and wasserstein distance
    ut.plot_and_save(G_losses, "Generator Loss during training",
                     c.save_results, "Generator_loss")
    ut.plot_and_save(D_losses, "Discriminator Loss during training",
                     c.save_results, "Discriminator_loss")
    ut.plot_and_save(epsilons, "Epsilon over epochs",
                     c.save_results, "Epsilons")

    epoch_end_time = time.time()

    duration = duration + (epoch_end_time - epoch_start_time)
    approx_time_to_finish = duration / (epoch + 1) * (c.num_epochs
                                                      - (epoch + 1))
    print("Training time for epoch ", epoch, ": ", (epoch_end_time
                                                    - epoch_start_time) / 60,
          " minutes = ", (epoch_end_time - epoch_start_time) / 3600, "hours.")
    print("Approximate time remaining for run to finish: ",
          approx_time_to_finish / 3600, " hours")

    gc.collect()
