from torch.utils.data import Dataset
import torch

import numpy as np
import gzip

import utils as ut
import config as c


# Custom dataset to read patches and labels and concat them
class GANDataset(Dataset):
    def __init__(self, patches_paths):
        self.patches_paths = patches_paths

    def __getitem__(self, index):
        patch_file = gzip.GzipFile(self.patches_paths[index], "r")
        patch = np.load(patch_file, allow_pickle=True)
        patch_file.close()
        patch_normalized = self.__itensity_normalize_one_volume__(patch)
        data = torch.FloatTensor(patch_normalized).unsqueeze(0)

        return data

    def __len__(self):
        return len(self.patches_paths)

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out
