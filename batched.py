import numpy as np

from offset_tensor import OffsetTensor
from periodic.wave import wavedec_period
from wavelet import Wavelet

def is_diag(m: np.ndarray):
    return np.all(m == np.diag(np.diagonal(m)))

def get_matrix_order(m: np.ndarray):
    Mp = m.copy()
    order = 1
    while not(is_diag(Mp)):
        order += 1
        Mp = Mp @ m
    return order, Mp

def wavedec_period_batched(data: OffsetTensor, w: Wavelet, level: int):
    order, Mp = get_matrix_order(w.M)
    div = level // order
    rem = level % order
    wave = []
    for l in range(div):
        wave.append(wavedec_period(data, w, order))
        #data = simple_downsample(data)
        w.M = Mp
        w.m = np.linalg.det(Mp)

