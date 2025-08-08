from math import ceil, floor

import numpy as np

from offset_matrix import OffsetMatrix
from wavelet import Wavelet

def OffsetMatrixConjugate_dummy(a_shape, a_offset):
    offset = np.array([-(a_offset[0] + a_shape[0] - 1), -(a_offset[1] + a_shape[1] - 1)])
    return a_shape, offset


def downsample_dummy(shape, offset, M: np.ndarray):
    Minv = np.linalg.inv(M)
    x1 = Minv @ np.array([offset[0], offset[1]])
    x2 = Minv @ np.array([offset[0] + shape[0] - 1, offset[1]])
    x3 = Minv @ np.array([offset[0], offset[1] + shape[1] - 1])
    x4 = Minv @ np.array([offset[0] + shape[0] - 1, offset[1] + shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))
    downsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    return downsampled.matrix.shape, downsampled.offset


def upsample_dummy(a_shape, a_offset, M: np.ndarray):
    x1 = M @ np.array([a_offset[0], a_offset[1]])
    x2 = M @ np.array([a_offset[0] + a_shape[0] - 1, a_offset[1]])
    x3 = M @ np.array([a_offset[0], a_offset[1] + a_shape[1]-1])
    x4 = M @ np.array([a_offset[0] + a_shape[0]  - 1, a_offset[1] + a_shape[1] - 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    return upsampled.matrix.shape, upsampled.offset


def convolve_dummy(shape, offset, mask_shape, mask_offset):
    new_offset = np.array([offset[0] + mask_offset[0], offset[1] + mask_offset[1]])
    new_shape = np.array([shape[0] + (mask_shape[0] - 1), shape[1] + (mask_shape[1] - 1)])
    return new_shape, new_offset


def transition_dummy(shape, offset, mask_shape, mask_offset, M):
    (mask_shape, mask_offset) = OffsetMatrixConjugate_dummy(mask_shape, mask_offset)
    (shape, offset) = convolve_dummy(shape, offset, mask_shape, mask_offset)
    return shape, offset


def subdivision_dummy(matrix_shape, matrix_offset, mask_shape, mask_offset, M: np.ndarray):
    (shape, offset) = upsample_dummy(matrix_shape, matrix_offset ,M)
    (offset_, dummy_) = convolve_dummy(shape, offset, mask_shape, mask_offset)
    return offset_, dummy_

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
    shape_, offset_ = transition_dummy(data_shape, data_offset, ref_mask_shape, ref_mask_offset, cur_M)
    details.append([[shape_, offset_]])
    details.reverse()

    return details

