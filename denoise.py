import itertools

import numpy as np
import scipy

def est_sigma(details):
    denom = scipy.stats.norm.ppf(0.995)
    sigma = np.median(np.abs(details)) / denom
    return sigma

def bayes_thresh(detail, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(detail * detail)
    eps = np.finfo(detail.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh

def apply_bayes_thresh(wavecoef):
    details = wavecoef[1:]
    details_flat = np.array(list(itertools.chain(*itertools.chain(*wavecoef[1:]))))
    sigma = est_sigma(details_flat)
    var = sigma * sigma
    denoised = [wavecoef[0]] + [[bayes_thresh(c, var) for c in level_coef] for level_coef in details]
    return denoised


def universal_thresh(img, wavecoef):
    """Universal threshold used by the VisuShrink method"""
    details = np.array(list(itertools.chain(*itertools.chain(*wavecoef[1:]))))
    sigma = est_sigma(details)
    return sigma * np.sqrt(2 * np.log(img.tensor.size))

