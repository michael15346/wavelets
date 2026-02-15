import itertools

import numpy as np
import scipy

def est_sigma(details):
    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(details)) / denom
    return sigma

def bayes_thresh(coef):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    details = np.array(list(itertools.chain(*itertools.chain(*coef[1:]))))
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    sigma = est_sigma(details)
    var = sigma * sigma
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def universal_thresh(img, coef):
    """Universal threshold used by the VisuShrink method"""
    details = np.array(list(itertools.chain(*itertools.chain(*coef[1:]))))
    sigma = est_sigma(details)
    return sigma * np.sqrt(2 * np.log(img.tensor.size))

