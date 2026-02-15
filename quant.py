import itertools

import numpy as np

from dummy.wave import wavedec_periodic_dummy
from wavelet import Wavelet


def encode_kmeans(coef: list, n_cluster = 256):
    flat_coef = np.array(list(coef[0] - 128) + list(itertools.chain(*itertools.chain(*coef[1:]))))
    clusters = fast1dkmeans.cluster(flat_coef, n_cluster, method='binary-search-interpolation')
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=flat_coef)
    counts = np.bincount(inverse)
    centroids = sums / counts
    return centroids, clusters

def decode_kmeans(centroids: np.ndarray, clusters: np.ndarray, w: Wavelet, original_shape, level: int):
    coef_shapes = wavedec_periodic_dummy(original_shape, np.zeros_like(original_shape), w, level)
    flat_restored = centroids[clusters]
    restored = list()
    restored.append(flat_restored[:coef_shapes[0]] + 128)
    idx = coef_shapes[0]
    for c in coef_shapes[1:]:
        restored.append([])
        for cc in c:
            restored[-1].append(flat_restored[idx:idx + cc])
            idx += cc
    return restored

def entropy(coef: list):
    val, counts = np.unique(coef, return_counts=True)
    prob = counts / len(coef)
    return -(prob * np.log2(prob)).sum()


def encode_uniform(coef: list, n_cluster=16):

    flat_coef = np.array( list(itertools.chain(*itertools.chain(*coef[1:]))))
    downs_coef = np.array(list(coef[0] - 128))
    n_min = np.min(flat_coef)
    downs_min = np.min(downs_coef)
    downs_max = np.max(downs_coef)
    n_max = np.max(flat_coef)
    quantized = np.rint(np.clip((flat_coef - n_min) / (n_max - n_min) * (n_cluster - 1), 0, n_cluster - 1)).astype(int)
    quantized_downs = np.rint(np.clip((downs_coef - downs_min) / (downs_max - downs_min) * (n_cluster - 1), 0, n_cluster - 1)).astype(int)
    return quantized, quantized_downs, n_min, n_max, downs_min, downs_max

def decode_uniform(quantized: np.ndarray, quantized_downs: np.ndarray, n_min: float, n_max: float, downs_min: float, downs_max: float, n_cluster: int,w: Wavelet, original_shape, level: int):
    flat_restored = np.concatenate((quantized_downs * (downs_max - downs_min) / (n_cluster - 1) + downs_min,
                                   quantized * (n_max - n_min) / (n_cluster - 1) + n_min))
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

def hard_threshold(array, threshold):
    return np.where(np.abs(array) >= threshold, array, 0)

def soft_threshold(array, threshold):
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0)

def apply_threshold(wavecoef: list, threshold: float):
    downs_coef, flat_coef, coef_lens = wavecoef_to_array(wavecoef)
    thres_coef = hard_threshold(flat_coef, threshold)
    thres_wavecoef = array_to_wavecoef(downs_coef, thres_coef, coef_lens)

    #thres_wavecoef.insert(0, flat_coef[0])

    return thres_wavecoef, threshold

def apply_threshold_quantile(wavecoef: list, quantile: float = 0.01):
    downs_coef, flat_coef, _ = wavecoef_to_array(wavecoef)
    threshold = np.quantile(np.abs(flat_coef), quantile)
    return apply_threshold(wavecoef, threshold)

def get_wavecoef_shape(wavecoef):
    lengths = []
    for item in wavecoef:
        # if isinstance(item, np.ndarray):
        #     # Если элемент - массив, добавляем его длину
        #     lengths.append(len(item))
        # elif isinstance(item, list):
            # Если элемент - список, обрабатываем каждый подэлемент
            w_lengths = []
            for sub_item in item:
                if isinstance(sub_item, np.ndarray):
                    w_lengths.append(len(sub_item))
            lengths.append(w_lengths)
    return lengths

def wavecoef_to_array(wavecoef):
    coef_lens = get_wavecoef_shape(wavecoef[1:])
    flat_coef = np.concatenate([np.array(list(itertools.chain(*itertools.chain(*wavecoef[1:]))))])
    downs_coef = wavecoef[0]
    return downs_coef, flat_coef, coef_lens

def array_to_wavecoef(downs_coef, flat_coef, coef_lens):
    wavecoef = [downs_coef]
    idx = 0
    for wave_lens in coef_lens:
        wavecoef.append([])
        for wave_len in wave_lens:
            wavecoef[-1].append(flat_coef[idx:idx + wave_len])
            idx += wave_len
    return wavecoef
