import numpy as np

from offset_matrix import OffsetMatrix
from periodic.operators import subdivision_period, transition_period
from vector.operators import transition_vector, subdivision_vector
from wavelet import Wavelet
from classic.operators import subdivision


def wavedec_period(data: OffsetMatrix, w: Wavelet, level: int):
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
    #masks[-1].append(ref_mask)

    details = []
    cur_M = np.eye(w.M.shape[0], dtype=int)
    for mask in masks:
        cur_M @= w.M
        tmp_list = list()
        for m in mask:
            tmp_list.append(transition_period(data, m, cur_M.copy()))
        details.append(tmp_list)
        #details.append(list(map(
        #    transition, [data] * len(mask), mask, [cur_M] * len(mask))))
    details.append(transition_period(data, ref_mask, cur_M))
    details.reverse()


    return details


def waverec_period(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):


    a = c[0]
    d = c[1:]
    d.reverse()
    res = OffsetMatrix(np.zeros((1, 1)), np.array([0, 0]))
    m = w.m
    wmasks = [OffsetMatrix(wmask.matrix * m, wmask.offset) for wmask in w.g]
    cur_M = w.M.copy()
    for i, di in enumerate(d):
        for j, dij in enumerate(di):
            res += subdivision_period(dij, wmasks[j], cur_M, original_shape, original_offset)
            wmasks[j] = subdivision(wmasks[j], w.h, w.M)
            wmasks[j].matrix = wmasks[j].matrix * m
            cur_M @= w.M

    mask_h = OffsetMatrix(w.h.matrix * m, w.h.offset)
    cur_M = w.M.copy()
    for i in range(len(d)-1):
        mask_h = subdivision(mask_h, w.h, w.M)
        mask_h.matrix = mask_h.matrix * m
        cur_M @= w.M

    res += subdivision_period(a, mask_h, cur_M, original_shape, original_offset)


    #res = res.matrix[*tuple(map(slice, tuple(-res.offset), tuple(-res.offset+original_shape)))]
    res.matrix = res.matrix[-res.offset[0]:-res.offset[0] + original_shape[0], -res.offset[1] :-res.offset[1] + original_shape[1]]
    return res



