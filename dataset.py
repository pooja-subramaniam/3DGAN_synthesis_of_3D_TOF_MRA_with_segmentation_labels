from torch.utils.data import Dataset
import torch

import numpy as np
import gzip

import utils as ut
import config as c


# Custom dataset to read patches and labels and concat them
class GANDataset(Dataset):
    def __init__(self, patches_paths, labels_paths):
        self.patches_paths = patches_paths
        self.labels_paths = labels_paths

    def __getitem__(self, index):
        patch_file = gzip.GzipFile(self.patches_paths[index], "r")
        patch = np.load(patch_file)
        patch_file.close()
        patch_normalized = ut.normalize(patch)  # normalize to [+1,-1] range

        if c.nc == 2:
            label_file = gzip.GzipFile(self.labels_paths[index], "r")
            label = np.load(label_file)
            label_file.close()

            data_np = np.stack((patch_normalized, label), axis=0)
            data = torch.FloatTensor(data_np)

        else:
            data = torch.FloatTensor(patch_normalized).unsqueeze(0)
        return data

    def __len__(self):
        if len(self.patches_paths) != len(self.labels_paths):
            raise Exception("Number of patches and labels are different")
        return len(self.patches_paths)
