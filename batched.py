from copy import deepcopy

import numpy as np

from offset_tensor import OffsetTensor
from periodic.wave import create_dwt_fb, transition_period_fastest, \
    subdivision_period_fastest, waverec_period_fastest, convolve_period, wavedec_period_fastest
from vector.operators import get_pad_up_to
from wavelet import Wavelet

def is_diag(M: np.ndarray):
    return np.all(M == np.diag(np.diagonal(M)))

def get_matrix_order(M: np.ndarray, max_level):
    Mp = M.copy()
    order = 1
    while not(is_diag(Mp)):
        if order == max_level:
            print("Max levels learched without diagonal")
            print(M)
            return None, None
        order += 1
        Mp = Mp @ M

    return order, Mp


def transition_batched_simple_downs(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = mask.conjugate()
    return simple_downsample(convolve_period(a, mask), M)

def subdivision_batched_simple_ups(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray, original_shape, original_offset):
    u = simple_upsample(a,M)
    c = convolve_period(u, mask)
    return c


def wavedec_period_batch(data: OffsetTensor, w: Wavelet, level: int):

    masks, Ms = create_dwt_fb(w.hdual, w.gdual, w.M, level)

    dwt_coefs = []
    for cur_level, gmasks, cur_M in zip(range(level, 0, -1), masks[1:], Ms[1:]):
        tmp_list = []
        mod = w.m ** (cur_level / 2)
        for gmask in gmasks:
            tmp_list.append(transition_period_fastest(data * mod, gmask, cur_M))
        dwt_coefs.append(tmp_list)
    dwt_coefs.insert(0, transition_batched_simple_downs(data * (w.m ** (level / 2)), masks[0], Ms[0]))

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
        mod = m ** (cur_level / 2)
        for wave_mask, d_coef in zip(wave_masks, d_coefs):
            tmp = subdivision_period_fastest(d_coef, wave_mask, cur_M, shape, original_offset)
            res += OffsetTensor(tmp.tensor * mod, tmp.offset)

    tmp = subdivision_batched_simple_ups(c[0], masks[0], Ms[0], shape, original_offset)
    res += OffsetTensor(tmp.tensor * (m ** (level / 2)), tmp.offset)

    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, res.offset))

    res.tensor = res.tensor[slices]
    return res


def simple_downsample(data: OffsetTensor, M):
    shape = data.tensor.shape
    d = len(shape)
    M_is_diag = is_diag(M)
    M_diag = np.diag(M) if M_is_diag else np.diag(M[:, ::-1])

    slices = []
    for idx, value in enumerate(M_diag):
        if value > 0:
            slices.append(slice(0, None, value))
        else:
            slices.append(slice(shape[idx] + value, None, value))

    res_tensor = data.tensor[tuple(slices)] if M_is_diag else data.tensor[tuple(slices)].T
    roll_axis = np.arange(d) if M_is_diag else np.arange(d)[::-1]
    res_tensor = np.roll(res_tensor, tuple(M_diag < 0), axis=roll_axis)

    return OffsetTensor(res_tensor, data.offset)


def simple_upsample(data: OffsetTensor, M):
    shape = data.tensor.shape
    d = len(shape)
    M_is_diag = is_diag(M)
    M_diag = np.diag(M) if M_is_diag else np.diag(M[:, ::-1])
    roll_axis = np.arange(d) if M_is_diag else np.arange(d)[::-1]

    data.tensor = np.roll(data.tensor, tuple(-1 * (M_diag < 0)), axis=roll_axis)
    new_shape = np.abs(M @ shape).astype(int)
    upsampled = OffsetTensor(np.zeros(new_shape), np.zeros_like(shape))

    slices = []
    for idx, value in enumerate(M_diag):
        if value > 0:
            slices.append(slice(0, None, value))
        else:
            slices.append(slice(new_shape[idx] + value, None, value))

    upsampled.tensor[tuple(slices)] = data.tensor if M_is_diag else data.tensor.T

    return upsampled
def wavedec_period_batched(data: OffsetTensor, w: Wavelet, level: int):
    order, Mp = get_matrix_order(w.M, level)
    if not order:
        div = 0
        rem = level
    else:
        div = level // order
        rem = level % order
    shape = np.array(data.tensor.shape)
    padded_shape = get_pad_up_to(shape, np.linalg.matrix_power(w.M, level))
    pad_width = np.array([np.zeros_like(shape), padded_shape - shape], dtype=int).T
    data_padded = OffsetTensor(np.pad(data.tensor, pad_width, mode="wrap"), data.offset)
    data = data_padded
    wave = []
    for l in range(div):
        res = wavedec_period_batch(data, w, order)
        wave.extend(res[1:])
        data = res[0]
    if rem > 0:
        res = wavedec_period_fastest(data, w, rem)
        wave.extend(res[1:])
    wave.insert(0, res[0])
    return wave

def waverec_period_batched(c: list, w: Wavelet, original_shape):

    level = len(c[1:])
    order, Mp = get_matrix_order(w.M, level)
    if not order:
        div = 0
        rem = level
        order = level
    else:
        div = level // order
        rem = level % order
        Mpd = Mp.diagonal()
    shapes = []

    shape = get_pad_up_to(original_shape, np.linalg.matrix_power(w.M, level))
    for l in range(div):
        shapes.append(deepcopy(shape))
        shape //= np.abs(Mpd)
    img = c[0]
    if rem > 0:
        coef = [img] + c[-rem:]
        img = waverec_period_fastest(coef, w, shape)

    for i, idx in enumerate(range(len(c) - rem, 1, -order)):
        cc = c[idx-order:idx]
        coef = [img] + cc
        img = waverec_period_batch(coef, w, shapes[-i-1])
    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, img.offset))

    img.tensor = img.tensor[slices]
    return img

