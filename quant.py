import itertools

import fast1dkmeans
import numpy as np

def roundtrip(coef: list, n_cluster = 8):
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
    idx = 0
    restored.append(flat_restored[:coef[0].size])
    idx = coef[0].size
    for c in coef[1:]:
        restored.append([])
        for cc in c:
            restored[-1].append(flat_restored[idx:idx + cc.size])
            idx += cc.size
    return restored