import numpy as np

from dummy.operators import subdivision_dummy, transition_dummy
from wavelet import Wavelet


def wavedec_multilevel_at_once_dummy(data_shape, data_offset, w: Wavelet, level: int):
    mask = [[ww.matrix.shape, ww.offset] for ww in w.gdual]
    masks = [mask]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_mask_shape = cur_mask.matrix.shape
            cur_mask_offset = cur_mask.offset
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask_shape, cur_mask_offset = subdivision_dummy(w.hdual.matrix.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
                cur_M @= w.M
            wave_mask_shape, wave_mask_offset = subdivision_dummy(gdual.matrix.shape, gdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
            gmasks.append([wave_mask_shape, wave_mask_offset])
        masks.append(gmasks)
    # !!!
    if level > 1:
        ref_mask_shape, ref_mask_offset = subdivision_dummy(w.hdual.matrix.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
    else:
        ref_mask_shape, ref_mask_offset = w.hdual.matrix.shape, w.hdual.offset
    details = []
    cur_M = np.eye(w.M.shape[0], dtype=int)
    for mask in masks:
        cur_M @= w.M
        tmp_list = list()
        for m in mask:
            shape_, offset_ = transition_dummy(data_shape, data_offset, m[0], m[1], cur_M.copy())
            tmp_list.append([shape_, offset_])
        details.append(tmp_list)
        #details.append(list(map(
        #    transition, [data] * len(mask), mask, [cur_M] * len(mask))))
    shape_, offset_ = transition_dummy(data_shape, data_offset, ref_mask_shape, ref_mask_offset, cur_M)
    details.append([[shape_, offset_]])
    details.reverse()

    return details

