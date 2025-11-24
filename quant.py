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

# def uniform_entropy(coef: list, n_cluster = 256):
#     flat_coef = np.array(list(coef[0]) + list(itertools.chain(*itertools.chain(*coef[1:]))))
#     print("before cluster")
#     n_min = np.min(flat_coef)
#     n_max = np.max(flat_coef)
#     quantized = np.clip(np.round((flat_coef - n_min) / n_max * (n_cluster - 1)),0, n_cluster - 1).astype(int)
#     val, counts = np.unique(quantized, return_counts=True)
#     prob = counts / len(quantized)
#     return -(prob * np.log2(prob)).sum()

def encode_uniform(coef: list, quantile: float = 0.01):

    flat_coef = np.array( list(itertools.chain(*itertools.chain(*coef[1:]))))
    downs_coef = np.array(list(coef[0] - 128))
    threshold = np.quantile(np.abs(downs_coef), quantile)
    idx = np.abs(flat_coef) >= threshold
    flat_coef[np.abs(flat_coef) < threshold] = 0
    flat_coef[idx] -= threshold * np.sign(flat_coef[idx])
    threshold_downs = np.quantile(np.abs(downs_coef), quantile)
    idx_downs = np.abs(downs_coef) >= threshold_downs
    downs_coef[np.abs(downs_coef) < threshold_downs] = 0
    downs_coef[idx_downs] -= threshold_downs * np.sign(downs_coef[idx_downs])
    return flat_coef, downs_coef

def decode_uniform(flat_coef: np.ndarray, downs_coef: np.ndarray, w: Wavelet, original_shape, level: int):
    flat_restored = np.concatenate((downs_coef, flat_coef))
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
