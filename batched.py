from copy import deepcopy

import numpy as np

from offset_tensor import OffsetTensor
from periodic.wave import wavedec_period_fastest, waverec_period_fastest
from wavelet import Wavelet

def is_diag(M: np.ndarray):
    return np.all(M == np.diag(np.diagonal(M)))

def get_matrix_order(m: np.ndarray):
    Mp = m.copy()
    order = 1
    while not(is_diag(Mp)):
        order += 1
        Mp = Mp @ m
    return order, Mp


def simple_downsample(data, M):
    if not is_diag(M):
        raise ValueError("M is not diagonal")

    if M.shape[0] != data.ndim:
        raise ValueError("M's size must match the number of array dimensions.")

    indices = []

    for axis, m in enumerate(M.diagonal()):
        if m == 0:
            raise ValueError("Step size cannot be zero.")

        size = data.shape[axis]
        new_length = (size + abs(m) - 1) // abs(m)

        idx = (np.arange(new_length) * m) % size
        indices.append(idx)

    return OffsetTensor(data[np.ix_(*indices)], offset=np.zeros_like(data.shape))

def wavedec_period_batched(data: OffsetTensor, w: Wavelet, level: int):
    order, Mp = get_matrix_order(w.M)
    div = level // order
    rem = level % order
    wave = []
    wmod = deepcopy(w)
    for l in range(div):
        res = wavedec_period_fastest(data, w, order)
        wave.append(res[1:])
        data = simple_downsample(res[0], Mp)
    if rem > 0:
        wmod.M = np.linalg.matrix_power(w.M, rem)
        wmod.m = np.linalg.det(wmod.M)
        res = wavedec_period_fastest(data, wmod, rem)
        wave.append(res[1:])
    wave.append([res[0]])
    return wave

def waverec_period_batched(c: list, w: Wavelet, original_shape):
    if len(c) > 1:
        div = len(c[0])
        rem = len(c[-1])
    else:
        div = 0
        rem = len(c[0])
    order, Mp = get_matrix_order(w.M)
    Mpd = Mp.diagonal()
    wmod = deepcopy(w)
    shapes = []
    shape = original_shape
    for l in range(div):
        shapes.append(deepcopy(shape))
        shape //= np.abs(Mpd)
    img = c[-1]
    start = 0
    if rem > 0:
        Mrem = np.linalg.matrix_power(w.M, rem)
        shapes.append(shape // np.abs(Mrem.diagonal()))
        coef = img + c[-2]
        wmod.M = Mrem
        wmod.m = np.linalg.det(wmod.M)
        img = waverec_period_fastest(coef, wmod, shapes[-1])
        start += 1

    wmod.M = Mp
    wmod.m = np.linalg.det(wmod.M)
    for i, cc in enumerate(reversed(c[:-1 - start])):
        coef = img + cc
        img = waverec_period_fastest(coef, wmod, shapes[-i-1 - start])
    return img[0]

