import numpy as np


def OffsetMatrixConjugate_dummy(a_shape, a_offset):
    offset = np.array([-(a_offset[0] + a_shape[0] - 1), -(a_offset[1] + a_shape[1] - 1)])
    return a_shape, offset
