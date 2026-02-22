import skimage.util


def gen_gaussian_noise(img, var=.01):
    return skimage.util.random_noise((img - 128) / 256, mode='gaussian', var=var) * 256 + 128

def gen_snp_noise(img, amount=.05):
    return skimage.util.random_noise((img - 128) / 256, mode='s&p', amount=amount) * 256 + 128