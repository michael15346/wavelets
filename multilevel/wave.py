import numpy as np

from classic.wave import subdivision
from dummy.wave import wavedec_multilevel_at_once_dummy
from offset_matrix import OffsetTensor
from vector.operators import transition_vector, subdivision_vector
from wavelet import Wavelet


def wavedec_multilevel_at_once(data: OffsetTensor, w: Wavelet, level: int):
    masks = [list(w.gdual)]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask = subdivision(w.hdual, cur_mask, cur_M)
                cur_M @= w.M
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
        cur_M @= w.M
        tmp_list = list()
        for m in mask:
            tmp_list.append(transition_vector(data, m, cur_M.copy()))
        details.append(tmp_list)
    details.append(transition_vector(data, ref_mask, cur_M))
    details.reverse()


    return details


def waverec_multilevel_at_once(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):


    a = c[0]
    d = c[1:]
    og_s_o = wavedec_multilevel_at_once_dummy(original_shape, original_offset, w, len(d))
    d.reverse()
    res = OffsetTensor(np.zeros((1,) * len(original_shape)), np.zeros_like(original_offset))
    m = w.m
    wmasks = [OffsetTensor(wmask.tensor * m, wmask.offset) for wmask in w.g]
    cur_M = w.M.copy()
    for i, di in enumerate(d):
        for j, dij in enumerate(di):
            res += subdivision_vector(dij, wmasks[j], cur_M, og_s_o[len(d) - i][0][0], og_s_o[len(d) - i][0][1])
            wmasks[j] = subdivision(wmasks[j], w.h, w.M)
            wmasks[j].tensor = wmasks[j].tensor * m
            cur_M @= w.M

    mask_h = OffsetTensor(w.h.tensor * m, w.h.offset)
    cur_M = w.M.copy()
    for i in range(len(d)-1):
        mask_h = subdivision(mask_h, w.h, w.M)
        mask_h.tensor = mask_h.tensor * m
        cur_M @= w.M

    res += subdivision_vector(a, mask_h, cur_M, og_s_o[0][0][0], og_s_o[0][0][1])

    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, res.offset))

    res.tensor = res.tensor[slices]
    return res



