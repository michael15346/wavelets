import numpy as np

from classic.wave import subdivision
from db import SetOfDigitsFinder
from ezw.operators import subdivision_ezw, transition_ezw, gen_coords_ezw, step_coords_ezw, init_coords_ezw
from offset_tensor import OffsetTensor
from wavelet import Wavelet



def wavedec_ezw(data: OffsetTensor, w: Wavelet, level: int):
    shape = np.array(data.tensor.shape)
    pad_up_to = np.rint(w.m ** level).astype(int)
    padding = np.array([np.zeros_like(shape), np.ceil(shape / pad_up_to) * pad_up_to - shape], dtype=int).T
    data_padded = OffsetTensor(np.pad(data.tensor, padding, mode="symmetric"), data.offset)
    padded_shape = np.array(data_padded.tensor.shape)
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


    ezw_coords = init_coords_ezw(padded_shape, data_padded.offset, level, w.M)
    details = [transition_ezw(data_padded, ref_mask, ezw_coords)]
    digits = SetOfDigitsFinder(w.M)
    Mdigits = [w.M @ digits.T]
    for l in range(level - 2):
        Mdigits.append(w.M @ Mdigits[-1])
    tmp_list = list()
    for m in masks[-1]:
        tmp_list.append(transition_ezw(data_padded, m, ezw_coords))
    details.append(tmp_list)
    for level, mask in enumerate(reversed(masks[:-1])):
        ezw_coords = step_coords_ezw(padded_shape, ezw_coords, Mdigits[-level-1])
        tmp_list = list()
        for m in mask:
            tmp_list.append(transition_ezw(data_padded, m, ezw_coords))
        details.append(tmp_list)


    return details



def waverec_ezw(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):

    a = c[0]
    d = c[1:]
    level = len(d)
    shape = np.array(original_shape, dtype=int)
    #padding = np.array(w.m ** level - shape % (w.m ** level), dtype=int).T
    pad_up_to = np.rint(w.m ** level).astype(int)
    padding = np.array(np.ceil(shape / pad_up_to) * pad_up_to - shape, dtype=int).T
    padded_shape = shape + padding
    d.reverse()
    res = OffsetTensor(np.zeros((1,) * len(padded_shape)), np.zeros_like(original_offset))
    m = w.m
    wmasks = [OffsetTensor(wmask.tensor * m, wmask.offset) for wmask in w.g]
    cur_M = w.M.copy()
    ezw_coords = gen_coords_ezw(padded_shape, original_offset, level, w.M)
    for i, di in enumerate(d):
        for j, dij in enumerate(di):
            res += subdivision_ezw(dij, wmasks[j], ezw_coords[i], padded_shape, original_offset)
            wmasks[j] = subdivision(wmasks[j], w.h, w.M)
            wmasks[j].tensor = wmasks[j].tensor * m
        cur_M = cur_M @ w.M

    mask_h = OffsetTensor(w.h.tensor * m, w.h.offset)
    cur_M = w.M.copy()
    for i in range(len(d)-1):
        mask_h = subdivision(mask_h, w.h, w.M)
        mask_h.tensor = mask_h.tensor * m
        cur_M = cur_M @ w.M

    res += subdivision_ezw(a, mask_h, ezw_coords[-1], padded_shape, original_offset)

    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, res.offset))

    res.tensor = res.tensor[slices]
    return res



