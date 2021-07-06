import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pytorch_model_summary import summary

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

# get patches and labels paths
path_patches = glob.glob(c.dataroot+"train/patches/"+"*.gz")
path_labels = glob.glob(c.dataroot+"train/seg_labels/"+"*.gz")

dataset = GANDataset(path_patches, path_labels)

dataloader = data_utils.DataLoader(dataset, batch_size=c.batch_size,
                                   shuffle=True, num_workers=c.workers)

# Device selection
device = torch.device("cuda:" + str(c.cuda_n[0]) if (torch.cuda.is_available()
                                                     and
                                                     c.ngpu > 0) else "cpu")

# ####Create generator object##### #
netG = model.Generator().to(device)

# #### Create discriminator object #### #
netD = model.Discriminator().to(device)

if not c.continue_train:
    # Apply the weights_init function to randomly initialize all weights
    netG.apply(model.weights_init)
    # Apply the weights_init function to randomly initialize all weights
    netD.apply(model.weights_init)

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

# print and save config params in trials.csv
print("Configuration for the run: \n")
print(dict(zip(c.list_config_names, c.list_config)))
ut.save_config(c.save_config, c.list_config_names, c.list_config)

if c.continue_train:
    saved_params_dict = torch.load(c.saved_model_path,
                                   map_location=lambda storage, loc: storage)

    # Load Generator
    netG.load_state_dict(saved_params_dict['Generator_state_dict'])
    optimizerG.load_state_dict(saved_params_dict['OptimizerG_state_dict'])

    # Load Discriminator
    netD.load_state_dict(saved_params_dict['Discriminator_state_dict'])
    optimizerD.load_state_dict(saved_params_dict['OptimizerD_state_dict'])

    # Load lists with last values to keep track of progress
    G_losses = np.load(c.save_results + 'G_losses.npy')[
        :c.epoch_num_to_continue+1].tolist()
    D_losses = np.load(c.save_results + 'D_losses.npy')[
        :c.epoch_num_to_continue+1].tolist()
    Wasserstein_D = np.load(c.save_results + 'Wasserstein_D.npy')[
        :c.epoch_num_to_continue+1].tolist()
    start_epoch = c.epoch_num_to_continue+1

else:
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    Wasserstein_D = []
    start_epoch = 1

iters = 0
duration = 0

# Training Loop

print("Starting Training Loop...")

# Scaler that tracks the scaling of gradients
# only used when mixed precision is used
scaler = torch.cuda.amp.GradScaler(enabled=c.use_mixed_precision)

# For each epoch
for epoch in range(start_epoch, c.num_epochs+1):
    epoch_start_time = time.time()
    # For each batch in the dataloader
    errD_iter = []
    errG_iter = []
    Wasserstein_D_iter = []
    batch_start_time = time.time()

    for i, data in enumerate(dataloader, 1):
        # for each iteration in the epoch
        errD_disc_iter = []
        Wasserstein_D_disc_iter = []
        batch_duration = 0

        # Format batch of real data
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)

        # ########################## #
        # (1) Update D network n_disc times:
        # with wasserstein_loss gradient penalty
        # ######################### #

        # updating the critic n_disc number of times
        # before 1 update of generator - with 3D this is set to 1
        for k in range(c.n_disc):
            netD.zero_grad()
            netG.zero_grad()

            # For training with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, c.nz, 1, 1, 1, device=device)

            # Training within the mixed-precision autocast - enabled/disabled
            with torch.cuda.amp.autocast(enabled=c.use_mixed_precision):

                # Train with real batch
                errD_real = netD(real_cpu)
                # Calculate loss on all-real batch
                errD_real = -errD_real.view(-1).mean()

                # Generate fake image batch with G
                fake = netG(noise)

                # Train with fake batch
                errD_fake = netD(fake.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = errD_fake.view(-1).mean()

                with torch.cuda.amp.autocast(enabled=False):
                    # get epsilon value from uniform distribution
                    eps = torch.rand(1).item()
                interpolate = eps * real_cpu + (1 - eps) * fake

                d_interpolate = netD(interpolate)

            # get gradient penalty
            gradient_penalty = ut.wasserstein_gradient_penalty(interpolate,
                                                               d_interpolate,
                                                               c.lambdaa,
                                                               scaler)
            # Calculate gradients for D in backward pass
            scaler.scale(errD_real).backward(retain_graph=True)
            scaler.scale(errD_fake).backward(retain_graph=True)
            scaler.scale(gradient_penalty).backward()

            errD = errD_fake.item() + errD_real.item() \
                + gradient_penalty.item()

            wasserstein = errD_fake.item() + errD_real.item()

            errD_disc_iter.append(errD)
            Wasserstein_D_disc_iter.append(wasserstein)

        scaler.step(optimizerD)
        del errD_fake
        del errD_real
        del wasserstein
        del gradient_penalty
        del errD
        del eps
        del interpolate
        del d_interpolate
        del fake
        del noise

        errD_disc_avg = np.mean(np.array(errD_disc_iter))
        Wasserstein_D_disc_avg = np.mean(np.array(Wasserstein_D_disc_iter))

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()

        noise = torch.randn(b_size, c.nz, 1, 1, 1, device=device)

        with torch.cuda.amp.autocast(enabled=c.use_mixed_precision):
            fake = netG(noise)

            output_fake = netD(fake)

            errG = -output_fake.view(-1).mean()

        # Calculate gradients for G
        scaler.scale(errG).backward()
        scaler.step(optimizerG)
        scaler.update()
        # update the iteration errors
        errD_iter.append(errD_disc_avg)
        errG_iter.append(errG.item())
        Wasserstein_D_iter.append(Wasserstein_D_disc_avg)

        del errG
        del errD_disc_avg
        del Wasserstein_D_disc_avg
        del real_cpu
        del noise
        del fake
        del output_fake
        iters += 1

        # print after every 100 batches
        if i % 100 == 0:
            print("[%d/%d] batches done!\n" % (i,
                                               len(dataset)//c.batch_size))
            batch_end_time = time.time()
            batch_duration = batch_duration + batch_end_time - batch_start_time
            print("Training time for", i, "batches: ", batch_duration / 60,
                  " minutes.")

    print(" End of Epoch %d \n" % epoch)

    # Output training stats after each epoch
    avg_errD = np.mean(np.array(errD_iter))
    avg_errG = np.mean(np.array(errG_iter))
    avg_Wasserstein_D = np.mean(np.array(Wasserstein_D_iter))

    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tWasserstein D: %.4f'
          % (epoch, c.num_epochs, avg_errD, avg_errG, avg_Wasserstein_D))

    # Save Losses and outputs for plotting later
    G_losses.append(avg_errG.item())
    D_losses.append(avg_errD.item())
    Wasserstein_D.append(avg_Wasserstein_D)

    if not os.path.exists(c.save_results):
        os.makedirs(c.save_results)

    np.save(c.save_results + 'G_losses.npy', np.asarray(G_losses))
    np.save(c.save_results + 'D_losses.npy', np.asarray(D_losses))
    np.save(c.save_results + 'Wasserstein_D.npy', np.asarray(Wasserstein_D))

    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fixed_fake = netG(fixed_noise).detach().cpu()

    sample_idx = [0, 1, 2, 3]

    for idx in sample_idx:
        # hard thresholding for visualisation
        sample = fixed_fake[idx].clone()
        if c.save_nifti:
            ut.convert_and_save_to_nifti(sample[0].to(
                                         dtype=torch.float32).numpy(),
                                         c.save_results
                                         + "fake_while_training_"
                                         "epoch_%d_sample_%d_patch.nii.gz"
                                         % (epoch, idx))
            if c.nc == 2:
                ut.convert_and_save_to_nifti(sample[1].to(
                                             dtype=torch.float32).numpy(),
                                             c.save_results
                                             + "fake_while_training_"
                                             "epoch_%d_sample_%d_label.nii.gz"
                                             % (epoch, idx))

    # save model parameters'
    if c.is_model_saved:
        if not os.path.exists(c.save_model):
            os.makedirs(c.save_model)
        if (epoch) % c.save_n_epochs == 0:
            torch.save({'Discriminator_state_dict': netD.state_dict(),
                        'Generator_state_dict': netG.state_dict(),
                        'OptimizerD_state_dict': optimizerD.state_dict(),
                        'OptimizerG_state_dict': optimizerG.state_dict(),
                        'Scaler_dict': scaler.state_dict()
                        }, c.save_model + "epoch_{}.pth".format(epoch))

    # plot and save G_loss, D_loss and wasserstein distance
    ut.plot_and_save(G_losses, "Generator Loss during training",
                     c.save_results, "Generator_loss")
    ut.plot_and_save(D_losses, "Discriminator Loss during training",
                     c.save_results, "Discriminator_loss")
    ut.plot_and_save(Wasserstein_D, "Wasserstein distance during training",
                     c.save_results, "Wasserstein distance")

    epoch_end_time = time.time()

    duration = duration + (epoch_end_time - epoch_start_time)
    approx_time_to_finish = duration / (epoch + 1) * (c.num_epochs
                                                      - (epoch + 1))
    print("Training time for epoch ", epoch, ": ", (epoch_end_time
                                                    - epoch_start_time) / 60,
          " minutes = ", (epoch_end_time - epoch_start_time) / 3600, "hours.")
    print("Approximate time remaining for run to finish: ",
          approx_time_to_finish / 3600, " hours")

    del errD_iter
    del errG_iter
    del Wasserstein_D_iter

    del avg_errD
    del avg_errG
    del avg_Wasserstein_D
    del fixed_fake
    del sample
