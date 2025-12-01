from itertools import product

import numpy as np

from offset_tensor import OffsetTensor
from vector.operators import get_pad_up_to
from wavelet import Wavelet
from utils import to_python_vect

def OffsetMatrixConjugate_dummy(a_shape, a_offset):
    slices = tuple(-(o + s - 1) for s, o in zip(a_shape, a_offset))
    offset = np.array(slices)
    return a_shape, offset


def downsample_dummy(shape, offset, M: np.ndarray):
    Minv = np.linalg.inv(M)
    choices = [(offset[i], offset[i] + shape[i] - 1) for i in range(len(shape))]
    corners = list(product(*choices))
    xs = Minv @ np.array(corners).T
    minc = np.ceil(np.min(xs, axis=1))
    maxc = np.floor(np.max(xs, axis=1))
    downsampled = OffsetTensor(np.zeros(np.rint(maxc - minc + 1).astype(int), dtype=np.float64), minc)
    return downsampled.tensor.shape, downsampled.offset

def downsample_vector_dummy(shape, offset, M: np.ndarray):
    Minv_pre = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
    m = round(np.abs(np.linalg.det(M)))
    slices = tuple(slice(o, o + s) for s, o in zip(shape, offset))
    lattice_coords = np.mgrid[slices].reshape(offset.shape[0], -1)
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = tuple(to_python_vect(lattice_coords.T[mask], offset))
    return len(lattice_coords[0])

def upsample_dummy(a_shape, a_offset, M: np.ndarray):
    choices = [(a_offset[i], a_offset[i] + a_shape[i] - 1) for i in range(len(a_shape))]
    corners = list(product(*choices))
    xs = M @ np.array(corners).T
    minc = np.ceil(np.min(xs, axis=1))
    maxc = np.floor(np.max(xs, axis=1))
    upsampled = OffsetTensor(np.zeros(maxc - minc + 1, dtype=np.float64), minc)
    return upsampled.tensor.shape, upsampled.offset


def convolve_dummy(shape, offset, mask_shape, mask_offset):
    new_offset = offset + mask_offset
    new_shape = np.array(shape) + np.array(mask_shape) - 1
    return new_shape, new_offset

def convolve_period_dummy(shape, offset, mask_shape):
    new_shape = np.array(shape)# + np.array(mask_shape) - 1

    return new_shape, np.zeros_like(offset)


def transition_dummy(shape, offset, mask_shape, mask_offset, M):
    (mask_shape, mask_offset) = OffsetMatrixConjugate_dummy(mask_shape, mask_offset)
    (shape, offset) = convolve_dummy(shape, offset, mask_shape, mask_offset)
    downsampled_shape, downsampled_offset = downsample_dummy(shape, offset, M)
    return downsampled_shape, downsampled_offset

def transition_period_dummy(shape, offset, mask_shape, mask_offset, M):
    (mask_shape, mask_offset) = OffsetMatrixConjugate_dummy(mask_shape, mask_offset)
    (shape, offset) = convolve_period_dummy(shape, offset, mask_shape)
    downsampled_shape = downsample_vector_dummy(shape, offset, M)
    return downsampled_shape


def subdivision_dummy(matrix_shape, matrix_offset, mask_shape, mask_offset, M: np.ndarray):
    (shape, offset) = upsample_dummy(matrix_shape, matrix_offset ,M)
    (offset_, dummy_) = convolve_dummy(shape, offset, mask_shape, mask_offset)
    return offset_, dummy_

def wavedec_multilevel_at_once_dummy(data_shape, data_offset, w: Wavelet, level: int):
    mask = [[ww.tensor.shape, ww.offset] for ww in w.gdual]
    masks = [mask]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_mask_shape = cur_mask.tensor.shape
            cur_mask_offset = cur_mask.offset
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask_shape, cur_mask_offset = subdivision_dummy(w.hdual.tensor.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
                cur_M @= w.M
            wave_mask_shape, wave_mask_offset = subdivision_dummy(gdual.tensor.shape, gdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
            gmasks.append([wave_mask_shape, wave_mask_offset])
        masks.append(gmasks)
    # !!!
    if level > 1:
        ref_mask_shape, ref_mask_offset = subdivision_dummy(w.hdual.tensor.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
    else:
        ref_mask_shape, ref_mask_offset = w.hdual.tensor.shape, w.hdual.offset
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

def wavedec_periodic_dummy(data_shape, data_offset, w: Wavelet, level: int):
    pad_up_to = get_pad_up_to(data_shape, w.M, level)
    padding = np.ceil(data_shape / pad_up_to) * pad_up_to - data_shape
    data_shape += padding
    masks = [[(gd.tensor.shape, gd.offset) for gd in w.gdual]]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_mask_shape = cur_mask.tensor.shape
            cur_mask_offset = cur_mask.offset
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask_shape, cur_mask_offset = subdivision_dummy(w.hdual.tensor.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
                cur_M @= w.M
            wave_mask_shape, wave_mask_offset = subdivision_dummy(gdual.tensor.shape, gdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
            gmasks.append([wave_mask_shape, wave_mask_offset])
        masks.append(gmasks)
    # !!!
    if level > 1:
        ref_mask_shape, ref_mask_offset = subdivision_dummy(w.hdual.tensor.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
    else:
        ref_mask_shape, ref_mask_offset = w.hdual.tensor.shape, w.hdual.offset

    details = []
    cur_M = np.eye(w.M.shape[0], dtype=int)
    for mask in masks:
        cur_M = cur_M @ w.M
        tmp_list = list()
        for m in mask:
            tmp_list.append(transition_period_dummy(data_shape, data_offset, m[0], m[1], cur_M.copy()))
        details.append(tmp_list)
    details.append(transition_period_dummy(data_shape, data_offset, ref_mask_shape, ref_mask_offset, cur_M))
    details.reverse()

    return details
