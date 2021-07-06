import numpy as np
import random
import matplotlib.pyplot as plt
import gzip
import os
import csv

import torch
import torch.autograd as autograd

import nibabel as nib

import config as c


# #################################################### #
# ####### data pre-processing helper functions ####### #
# #################################################### #

def normalize(x):
    """ Normalizes the image patches to be between -1 and +1

    input:
    x: n-dimensional array as input

    output: normalised n-dimensional array
    """
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def rescale_unet(x):
    """ Rescale the -1 to +1 ranged generated images to 0-255 range

    input:
    x: n-dimensional array as input

    output: rescaled n-dimensional array
    """
    return 255 * (x - x.min()) / (x.max() - x.min())


# ################################################## #
# ######### Set seed function ########### #
# ################################################## #

def set_all_seeds_as(seed):
    """ Sets random seeds of all modules for reproducibility

    input:
    seed: an integer to be set as seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ################################################## #
# ######### Print settings ########### #
# ################################################## #

def save_config(save_config_path, list_config_names, list_config):
    """ Save or append config file for parameter settings of different runs

    input:
    save_config_path: path where the file has to be saved
    list_config_names: header of the file of different parameters
    list_config: list of parameter values to be saved or appended
    """
    if not os.path.exists(save_config_path):
        with open(save_config_path, 'w') as file:
            fileWriter = csv.writer(file)
            fileWriter.writerow(list_config_names)
            fileWriter.writerow(list_config)
    else:
        with open(save_config_path, 'a') as file:
            fileWriter = csv.writer(file)
            fileWriter.writerow(list_config)


# ################################################## #
# ######### Wasserstein helper function ########### #
# ################################################## #

def wasserstein_gradient_penalty(interpolate, d_interpolate, lambdaa, scaler):
    """ Calculate gradient penalty from gradients using the formula

    input:
    interpolate: x_cap is randomly weighted average of a real and generated sample
    d_interpolate: output of critic with x_cap as input
    lambdaa: 10 from WGAN-GP literature
    scaler: scaling used for mixed precision

    output: rescaled n-dimensional array
    """

    grad_outputs = torch.ones_like(d_interpolate)
    scaled_gradients = autograd.grad(outputs=scaler.scale(d_interpolate),
                                     inputs=interpolate,
                                     grad_outputs=grad_outputs,
                                     create_graph=True,
                                     retain_graph=True,
                                     only_inputs=True)

    inv_scale = 1./scaler.get_scale()
    gradients = [p * inv_scale for p in scaled_gradients][0]
    with torch.cuda.amp.autocast(c.use_mixed_precision):
        gradient_norm = (gradients.norm(2) - 1) ** 2
        gradient_penalty = gradient_norm.mean() * lambdaa

    return gradient_penalty


# ################################################################ #
# ############ Plot and save graphs helper functions ############# #
# ################################################################ #

def plot_and_save(x, title, location_to_save, fname):
    """ Error and distance plots while training

    input:
    x: array to be plotted
    title: title of the plot
    location_to_save: to be saved
    fname: name of the file
    """
    plt.figure(figsize=(20, 7))
    plt.title(title)
    plt.plot(x)
    plt.xlabel("epoch")
    plt.ylabel(fname)
    plt.savefig(location_to_save + fname + ".png")
    plt.close()


# #################################################### #
# ####### Data post-processing helper functions ###### #
# #################################################### #

def convert_and_save_to_nifti(sample, location_to_save):
    converted = nib.Nifti1Image(sample, np.eye(4))
    converted.header.get_xyzt_units()
    converted.to_filename(location_to_save)  # Save as NiBabel file


def save_to_npy_gz(sample, location_to_save):
    write_file = gzip.GzipFile(location_to_save, "w")
    np.save(file=write_file, arr=sample)
    write_file.close()
