import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    a_flat = a.flatten()
    b_flat = b.flatten()
    se = 0.
    for i in range(a_flat.size):
        se += (a_flat[i] - b_flat[i]) ** 2

    return np.sqrt(se / a_flat.size)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    a_flat = a.flatten()
    return 20 * np.log10(np.max(a_flat) / rmse(a, b))
