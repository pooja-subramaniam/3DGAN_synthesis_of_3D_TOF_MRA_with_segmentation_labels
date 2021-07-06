import glob
import gzip
import numpy as np
from scipy.linalg import sqrtm

import random


def squash_features(dir):

    features = []
    i = 0
    for file in glob.glob(dir+'*.npy.gz'):
        f = gzip.GzipFile(file, "r")
        feature = np.load(f)
        f.close()
        # print(feature.shape)
        features.append(feature.flatten())
        i += 1
    # print(len(features))
    return np.asarray(random.sample(features, len(features)))
    # return np.asarray(features)


def calculate_fid(features_1, features_2):

    # calculate mean and covariance statistics
    mu1, sigma1 = features_1.mean(axis=0), np.cov(features_1, rowvar=False)
    mu2, sigma2 = features_2.mean(axis=0), np.cov(features_2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
