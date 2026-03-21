import itertools

import numpy as np

def hard_threshold(array, threshold):
    return np.where(np.abs(array) >= threshold, array, 0)

def soft_threshold(array, threshold):
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0)

def apply_threshold(wavecoef: list, threshold: float):
    downs_coef, flat_coef, coef_lens = wavecoef_to_array(wavecoef)
    thres_coef = hard_threshold(flat_coef, threshold)
    thres_wavecoef = array_to_wavecoef(downs_coef, thres_coef, coef_lens)

    return thres_wavecoef, threshold

def apply_soft_threshold(wavecoef: list, threshold: float):
    downs_coef, flat_coef, coef_lens = wavecoef_to_array(wavecoef)
    thres_coef = soft_threshold(flat_coef, threshold)
    thres_wavecoef = array_to_wavecoef(downs_coef, thres_coef, coef_lens)

    return thres_wavecoef, threshold

def apply_threshold_quantile(wavecoef: list, quantile: float = 0.01):
    downs_coef, flat_coef, _ = wavecoef_to_array(wavecoef)
    threshold = np.quantile(np.abs(flat_coef), quantile)
    return apply_threshold(wavecoef, threshold)

def get_wavecoef_shape(wavecoef):
    lengths = []
    for item in wavecoef:
        w_lengths = []
        for sub_item in item:
            if isinstance(sub_item, np.ndarray):
                w_lengths.append(sub_item.shape)
        lengths.append(w_lengths)
    return lengths

def wavecoef_to_array(wavecoef):
    coef_lens = get_wavecoef_shape(wavecoef[1:])
    flat_coef = np.concatenate(list(map(lambda x: x.reshape((-1,)), list(itertools.chain(*itertools.chain(*wavecoef[1:]))))), axis=None)
    downs_coef = wavecoef[0]
    return downs_coef, flat_coef, coef_lens

def array_to_wavecoef(downs_coef, flat_coef, coef_lens):
    wavecoef = [downs_coef]
    idx = 0
    for wave_lens in coef_lens:
        wavecoef.append([])
        for wave_shape in wave_lens:
            wave_len = np.prod(wave_shape)
            wavecoef[-1].append(flat_coef[idx:idx + wave_len].reshape(wave_shape))
            idx += wave_len
    return wavecoef
