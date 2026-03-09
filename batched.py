from copy import deepcopy

import numpy as np

from offset_tensor import OffsetTensor
from periodic.wave import create_dwt_fb, transition_period_fastest, \
    subdivision_period_fastest, waverec_period_fastest, convolve_period, wavedec_period_fastest
from vector.operators import get_pad_up_to
#from periodic.wave import waverec_period_fastest
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


def transition_batched_simple_downs(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = mask.conjugate()
    return simple_downsample(convolve_period(a, mask), M)


def simple_upsample(a, M, original_shape, original_offset):
    pass


def subdivision_batched_simple_ups(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray, original_shape, original_offset):
    u = simple_upsample(a,M, original_shape, original_offset)
    c = convolve_period(u, mask)
    return c


def wavedec_period_batch(data: OffsetTensor, w: Wavelet, level: int):

    masks, Ms = create_dwt_fb(w.hdual, w.gdual, w.M, level)

    dwt_coefs = []
    for gmasks, cur_M in zip(masks[1:], Ms[1:]):
        tmp_list = []
        for gmask in gmasks:
            tmp_list.append(transition_period_fastest(data, gmask, cur_M))
        dwt_coefs.append(tmp_list)
    dwt_coefs.insert(0, transition_batched_simple_downs(data, masks[0], Ms[0]))

    return dwt_coefs

def waverec_period_batch(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):
    if original_offset is None:
        original_offset = np.zeros_like(original_shape)

    level = len(c) - 1
    shape = np.array(original_shape, dtype=int)

    masks, Ms = create_dwt_fb(w.h, w.g, w.M, level)

    res = OffsetTensor(np.zeros((1,) * len(shape)), np.zeros_like(original_offset))
    m = w.m
    for cur_level, d_coefs, wave_masks, cur_M in zip(range(level, 0, -1), c[1:], masks[1:], Ms[1:]):
        for wave_mask, d_coef in zip(wave_masks, d_coefs):
            tmp = subdivision_period_fastest(d_coef, wave_mask, cur_M, shape, original_offset)
            res += OffsetTensor(tmp.tensor * (m ** cur_level), tmp.offset)

    tmp = subdivision_batched_simple_ups(c[0], masks[0], Ms[0], shape, original_offset)
    res += OffsetTensor(tmp.tensor * (m ** level), tmp.offset)

    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, res.offset))

    res.tensor = res.tensor[slices]
    return res


def simple_downsample(data, M):
    if not is_diag(M):
        raise ValueError("M is not diagonal")

    if M.shape[0] != data.tensor.ndim:
        raise ValueError("M's size must match the number of array dimensions.")

    indices = []

    for axis, m in enumerate(M.diagonal()):
        if m == 0:
            raise ValueError("Step size cannot be zero.")

        size = data.tensor.shape[axis]
        new_length = (size + abs(m) - 1) // abs(m)

        idx = (np.arange(new_length) * m) % size
        indices.append(idx)

    return OffsetTensor(data.tensor[np.ix_(*indices)], offset=np.zeros_like(data.tensor.shape))

def wavedec_period_batched(data: OffsetTensor, w: Wavelet, level: int):
    order, Mp = get_matrix_order(w.M)
    div = level // order
    rem = level % order
    shape = np.array(data.tensor.shape)
    padded_shape = get_pad_up_to(shape, np.linalg.matrix_power(w.M, level))
    pad_width = np.array([np.zeros_like(shape), padded_shape - shape], dtype=int).T
    data_padded = OffsetTensor(np.pad(data.tensor, pad_width, mode="wrap"), data.offset)
    data = data_padded
    div_list = []
    rem_list = []
    for l in range(div):
        res = wavedec_period_batch(data, w, order)
        div_list.append(res[1:])
        data = res[0]
    if rem > 0:
        res = wavedec_period_fastest(data, w, rem)
        rem_list.append(res[1:])
    wave = [div_list, rem_list, [res[0]]]
    return wave

def waverec_period_batched(c: list, w: Wavelet, original_shape):

    div = len(c[0])
    rem = len(c[1])
    order, Mp = get_matrix_order(w.M)
    level = order * div + rem
    Mpd = Mp.diagonal()
    wmod = deepcopy(w)
    shapes = []

    shape = get_pad_up_to(original_shape, np.linalg.matrix_power(w.M, level))
    for l in range(div):
        shapes.append(deepcopy(shape))
        shape //= np.abs(Mpd)
    img = c[-1]
    if rem > 0:
        coef = img + c[-2][0]
        img = waverec_period_fastest(coef, w, shape)

    for i, cc in enumerate(reversed(c[0])):
        coef = [img] + cc
        img = waverec_period_batch(coef, w, shapes[-i-1])
    return img[0]

