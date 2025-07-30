from math import ceil, floor

import numpy as np

from offset_matrix import OffsetMatrix


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
    return (downsampled.matrix.shape, downsampled.offset)


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
    return (upsampled.matrix.shape, upsampled.offset)

