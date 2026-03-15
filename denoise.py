import itertools

import numpy as np
import pywt
import scipy

from quant import soft_threshold


def est_sigma(details):
    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(details)) / denom
    return sigma

def est_sigma_1d(details):
    denom = scipy.stats.norm.ppf(0.75)
    lend = len(next(iter(details[-1].keys())))
    lastd = details[-1]['d' * lend]
    sigma = np.median(np.abs(lastd)) / denom
    return sigma

def bayes_thresh(detail, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(detail * detail)
    eps = np.finfo(detail.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return soft_threshold(detail, thresh)

def apply_bayes_thresh(wavecoef):
    details = wavecoef[1:]
    details_flat = np.concatenate(list(itertools.chain(*itertools.chain(*wavecoef[1:]))), axis=None)
    sigma = est_sigma(details_flat)
    var = sigma * sigma
    denoised = [wavecoef[0]] + [[bayes_thresh(c, var) for c in level_coef] for level_coef in details]
    return denoised

def apply_bayes_thresh_1d(wavecoef):
    details = wavecoef[1:]
    sigma = est_sigma_1d(details)
    var = sigma * sigma
    denoised = [wavecoef[0]] + [{k: bayes_thresh(v, var) for k, v in level_coef.items()} for level_coef in details]
    return denoised

def universal_thresh(img, wavecoef):
    """Universal threshold used by the VisuShrink method"""
    details = np.concatenate(list(itertools.chain(*itertools.chain(*wavecoef[1:]))), axis=None)
    sigma = est_sigma(details)
    return sigma * np.sqrt(2 * np.log(img.tensor.size))

def universal_thresh_1d(img, wavecoef):
    details = wavecoef[1:]
    sigma = est_sigma_1d(details)
    return sigma * np.sqrt(2 * np.log(img.size))

