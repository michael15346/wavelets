import itertools

import fast1dkmeans
import numpy as np

def roundtrip_kmeans(coef: list, n_cluster = 256):
    flat_coef = np.array(list(coef[0]) + list(itertools.chain(*itertools.chain(*coef[1:]))))
    print("before cluster")
    clusters = fast1dkmeans.cluster(flat_coef, n_cluster, method='binary-search-interpolation')
    print("after cluster")
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=flat_coef)
    counts = np.bincount(inverse)
    centroids = sums / counts
    flat_restored = centroids[clusters]
    restored = list()
    restored.append(flat_restored[:coef[0].size])
    idx = coef[0].size
    for c in coef[1:]:
        restored.append([])
        for cc in c:
            restored[-1].append(flat_restored[idx:idx + cc.size])
            idx += cc.size
    return restored

def uniform_entropy(coef: list, n_cluster = 256):
    flat_coef = np.array(list(coef[0]) + list(itertools.chain(*itertools.chain(*coef[1:]))))
    print("before cluster")
    n_min = np.min(flat_coef)
    n_max = np.max(flat_coef)
    quantized = np.clip(np.round((flat_coef - n_min) / n_max * (n_cluster - 1)),0, n_cluster - 1).astype(int)
    val, counts = np.unique(quantized, return_counts=True)
    prob = counts / len(quantized)
    return -(prob * np.log2(prob)).sum()

def uniform_roundtrip(coef: list, n_cluster = 16):
    flat_coef = np.array(list(coef[0]) + list(itertools.chain(*itertools.chain(*coef[1:]))))
    print("before cluster")
    n_min = np.min(flat_coef)
    n_max = np.max(flat_coef)
    quantized = np.clip(np.round((flat_coef - n_min) / n_max * (n_cluster - 1)),0, n_cluster - 1).astype(int)
    flat_restored = (quantized * n_max / (n_cluster - 1)) + n_min
    restored = list()
    restored.append(flat_restored[:coef[0].size])
    idx = coef[0].size
    for c in coef[1:]:
        restored.append([])
        for cc in c:
            restored[-1].append(flat_restored[idx:idx + cc.size])
            idx += cc.size
    return restored