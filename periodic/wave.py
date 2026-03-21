import numpy as np
import scipy

from classic.wave import subdivision
from offset_tensor import OffsetTensor
from vector.operators import get_pad_up_to, downsample_fastest, upsample_fastest
from wavelet import Wavelet

def transition_period_fastest(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = mask.conjugate()
    return downsample_fastest(convolve_period(a, mask), M)

def subdivision_period_fastest(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray, original_shape, original_offset):
    u = upsample_fastest(a,M, original_shape, original_offset)
    c = convolve_period(u, mask)
    return c

def convolve_period(a: OffsetTensor, b: OffsetTensor):
    new_offset = a.offset + b.offset
    origin = (-(np.array(b.tensor.shape) // 2)).tolist()

    new_tensor = scipy.ndimage.convolve(a.tensor.astype(np.float64), b.tensor.astype(np.float64),
                                        mode='wrap', origin=origin)
    # периодический сдвиг, чтобы (0,0) координата была в индексе (0,0)
    # origin в scipy.ndimage.convolve не умеет сдвигать больше, чем на пол-ядра относительно центра
    new_tensor = np.roll(new_tensor, tuple(new_offset.astype(np.int32)), axis=np.arange(len(a.offset)))
    return OffsetTensor(new_tensor, np.zeros_like(a.offset))

def create_dwt_fb(ref_mask, wave_masks, M, level: int):
    masks = [list(wave_masks)]
    Ms = [M]
    cur_M = M.copy()
    cur_mask = ref_mask

    for i in range(2, level+1):
        tmp_list = []
        for wave_mask in wave_masks:
            new_wave_mask = subdivision(wave_mask, cur_mask, cur_M)
            tmp_list.append(new_wave_mask)
        masks.insert(0, tmp_list)

        cur_mask = subdivision(ref_mask, cur_mask, cur_M)
        cur_M = cur_M @ M
        Ms.insert(0, cur_M)

    masks.insert(0, cur_mask)
    Ms.insert(0, cur_M)
    return masks, Ms

def wavedec_period_fastest(data: OffsetTensor, w: Wavelet, level: int):
    shape = np.array(data.tensor.shape)
    padded_shape = get_pad_up_to(shape, np.linalg.matrix_power(w.M, level))
    pad_width = np.array([np.zeros_like(shape), padded_shape - shape], dtype=int).T
    data_padded = OffsetTensor(np.pad(data.tensor, pad_width, mode="wrap"), data.offset)
    masks, Ms = create_dwt_fb(w.hdual, w.gdual, w.M, level)

    dwt_coefs = []
    for cur_level, gmasks, cur_M in zip(range(level, 0, -1), masks[1:], Ms[1:]):
        tmp_list = []
        mod = w.m ** (cur_level / 2)
        for gmask in gmasks:
            tmp_list.append(transition_period_fastest(data_padded * mod, gmask, cur_M))
        dwt_coefs.append(tmp_list)
    dwt_coefs.insert(0, transition_period_fastest(data_padded * (w.m ** (level / 2)), masks[0], Ms[0]))

    return dwt_coefs

def waverec_period_fastest(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):
    if original_offset is None:
        original_offset = np.zeros_like(original_shape)

    level = len(c) - 1
    shape = np.array(original_shape, dtype=int)
    padded_shape = get_pad_up_to(shape, np.linalg.matrix_power(w.M, level))

    masks, Ms = create_dwt_fb(w.h, w.g, w.M, level)

    res = OffsetTensor(np.zeros((1,) * len(padded_shape)), np.zeros_like(original_offset))
    m = w.m
    for cur_level, d_coefs, wave_masks, cur_M in zip(range(level, 0, -1), c[1:], masks[1:], Ms[1:]):
        mod = m ** (cur_level / 2)
        for wave_mask, d_coef in zip(wave_masks, d_coefs):
            tmp = subdivision_period_fastest(d_coef, wave_mask, cur_M, padded_shape, original_offset)
            res += OffsetTensor(tmp.tensor * mod, tmp.offset)

    tmp = subdivision_period_fastest(c[0], masks[0], Ms[0], padded_shape, original_offset)
    res += OffsetTensor(tmp.tensor * (m ** (level / 2)), tmp.offset)

    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, res.offset))

    res.tensor = res.tensor[slices]
    return res

