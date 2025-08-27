import numpy as np
import scipy

from classic.wave import subdivision
from offset_tensor import OffsetTensor
from vector.operators import upsample_vector, downsample_vector
from wavelet import Wavelet


def transition_period(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = mask.conjugate()
    return downsample_vector(convolve_period(a, mask), M)


def subdivision_period(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray, original_shape, original_offset):
    u = upsample_vector(a,M, original_shape, original_offset)
    c = convolve_period(u, mask)
    return c

def convolve_period(a: OffsetTensor, b: OffsetTensor):
    new_offset = a.offset + b.offset + np.ceil(np.array(b.tensor.shape) / 2) - 1
    new_tensor = scipy.ndimage.convolve(a.tensor.astype(np.float64), b.tensor.astype(np.float64), mode='wrap')
    # периодический сдвиг, чтобы (0,0) координата была в индексе (0,0)
    new_tensor = np.roll(new_tensor, tuple(new_offset.astype(np.int32)), axis=np.arange(len(a.offset)))
    return OffsetTensor(new_tensor, np.zeros_like(a.offset))

def wavedec_period(data: OffsetTensor, w: Wavelet, level: int):
    shape = np.array(data.tensor.shape)
    padding = np.array([np.zeros_like(shape), w.m ** level - shape % (w.m ** level)], dtype=int).T
    data_padded = OffsetTensor(np.pad(data.tensor, padding, mode="wrap"), data.offset)
    masks = [list(w.gdual)]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask = subdivision(w.hdual, cur_mask, cur_M)
                cur_M = cur_M @ w.M
            wave_mask = subdivision(gdual, cur_mask, cur_M)
            gmasks.append(wave_mask)
        masks.append(gmasks)
    # !!!
    if level > 1:
        ref_mask = subdivision(w.hdual, cur_mask, cur_M)
    else:
        ref_mask = w.hdual

    details = []
    cur_M = np.eye(w.M.shape[0], dtype=int)
    for mask in masks:
        cur_M = cur_M @ w.M
        tmp_list = list()
        for m in mask:
            tmp_list.append(transition_period(data_padded, m, cur_M.copy()))
        details.append(tmp_list)
    details.append(transition_period(data_padded, ref_mask, cur_M))
    details.reverse()


    return details

def waverec_period(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):

    a = c[0]
    d = c[1:]
    level = len(d)
    shape = np.array(original_shape, dtype=int)
    padding = np.array(w.m ** level - shape % (w.m ** level), dtype=int).T
    padded_shape = shape + padding
    d.reverse()
    res = OffsetTensor(np.zeros((1,) * len(padded_shape)), np.zeros_like(original_offset))
    m = w.m
    wmasks = [OffsetTensor(wmask.tensor * m, wmask.offset) for wmask in w.g]
    cur_M = w.M.copy()
    for i, di in enumerate(d):
        for j, dij in enumerate(di):
            res += subdivision_period(dij, wmasks[j], cur_M, padded_shape, original_offset)
            wmasks[j] = subdivision(wmasks[j], w.h, w.M)
            wmasks[j].tensor = wmasks[j].tensor * m
            cur_M = cur_M @ w.M

    mask_h = OffsetTensor(w.h.tensor * m, w.h.offset)
    cur_M = w.M.copy()
    for i in range(len(d)-1):
        mask_h = subdivision(mask_h, w.h, w.M)
        mask_h.tensor = mask_h.tensor * m
        cur_M = cur_M @ w.M

    res += subdivision_period(a, mask_h, cur_M, padded_shape, original_offset)

    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, res.offset))

    res.tensor = res.tensor[slices]
    return res



