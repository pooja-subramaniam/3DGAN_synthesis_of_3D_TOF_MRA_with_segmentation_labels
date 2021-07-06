import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import os
import numpy as np
import gzip
import glob

import config as c
from models import resnet
from dataset import GANDataset

# Device selection
device = torch.device("cuda:" + str(c.cuda_n[0]) if (torch.cuda.is_available() and
                                                     c.ngpu > 0) else "cpu")

def generate_model():
    model = resnet.resnet10(sample_input_W=c.image_size[0],
                sample_input_H=c.image_size[1],
                sample_input_D=c.image_size[2],
                num_seg_classes=2)
    return model

def get_features(data_loader, model, paths):
    model.eval()
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        path = paths[batch_id].split('/')[-1].split('.')[0]
        volume = batch_data
        volume = volume.to(device)
        with torch.no_grad():
            feature = model(volume).detach().cpu()

        if not isinstance(feature, np.ndarray):
            feature = np.asarray(feature)
        print(feature.shape)
        f = gzip.GzipFile(c.save_features_dir+path+'_em_features.npy.gz', "w")
        np.save(file=f, arr=feature)
        f.close()
 
    return


if __name__ == '__main__':

    # getting model
    print(f'getting pretrained model on {device} \n')
    checkpoint = torch.load(c.pretrained_model_path, map_location='cpu')
    print(checkpoint['state_dict'].keys())
    net = generate_model()
    print('Resnet model created \n')
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (c.ngpu >= 1):
        net = nn.DataParallel(net, c.cuda_n)
    net.load_state_dict(checkpoint['state_dict'])
    print('Resnet model loaded with pretrained weights')
    # data tensor
    print(f'Creating features from {c.dataroot}')
    path_patches = glob.glob(c.dataroot+"patches/"+"*.gz")
    dataset = GANDataset(path_patches)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    
    # embedded features
    get_features(data_loader, net, path_patches)
    
    print("All features saved \n")
