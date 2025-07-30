import numpy as np

from dummy.sample import upsample_dummy
from dummy.utils import OffsetMatrixConjugate_dummy


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
