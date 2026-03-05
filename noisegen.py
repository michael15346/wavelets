import numpy as np
import skimage.util


def gen_gaussian_noise(img, var=.01):
    return skimage.util.random_noise((img.astype(np.double) - 127.5) / 127.5, mode='gaussian', var=var) * 127.5 + 127.5

def gen_snp_noise(img, amount=.05):
    return skimage.util.random_noise((img.astype(np.double) - 127.5) / 127.5, mode='s&p', amount=amount) * 127.5 + 127.5