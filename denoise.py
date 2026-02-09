import itertools

import numpy as np
import scipy

from dummy.wave import wavedec_periodic_dummy

def est_sigma(coef):
    details = coef[1:]

    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(details)) / denom
    return sigma

def bayes_thresh(details):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def universal_thresh(img, sigma):
    """Universal threshold used by the VisuShrink method"""
    return sigma * np.sqrt(2 * np.log(img.size))

def encode_quant(coef: list, thresh, original_shape, w, level):

    flat_coef = np.array( list(itertools.chain(*itertools.chain(*coef[1:]))))
    downs_coef = np.array(list(coef[0] - 128))

    denoised = np.where(np.abs(flat_coef) > thresh, flat_coef, 0)
    flat_restored = np.concatenate((downs_coef,
                                   denoised))
    coef_shapes = wavedec_periodic_dummy(original_shape, np.zeros_like(original_shape), w, level)
    restored = list()
    restored.append(flat_restored[:coef_shapes[0]] + 128)
    idx = coef_shapes[0]
    for c in coef_shapes[1:]:
        restored.append([])
        for cc in c:
            restored[-1].append(flat_restored[idx:idx + cc])
            idx += cc
    return restored


def threshold(coef, original_shape, w, level, method="bayes_thresh"):
    details = coef[1:]
    if method == "bayes_thresh":
        thresh = bayes_thresh(details, sigma ** 2)
        return encode_quant(coef, thresh, original_shape, w, level)
    elif method == "universal_thresh":
        thresh = universal_thresh(details, sigma)
        return encode_quant(coef, thresh, original_shape, w, level)
    return None

