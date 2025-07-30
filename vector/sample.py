from math import ceil, floor

import numpy as np

from offset_matrix import OffsetMatrix
from utils import to_python_vect


def downsample_vector(a: OffsetMatrix, M: np.ndarray):
    Minv_pre = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
    m = round(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)

    x1 = Minv @ np.array([a.offset[0], a.offset[1]])
    x2 = Minv @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = Minv @ np.array([a.offset[0], a.offset[1] + a.matrix.shape[1] - 1])
    x4 = Minv @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1] + a.matrix.shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))

    downsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))

    lattice_coords = np.mgrid[a.offset[0]:a.offset[0] + a.matrix.shape[0],
                     a.offset[1]:a.offset[1] + a.matrix.shape[1]].reshape(2, -1)
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)
    return a.matrix[lattice_coords[0], lattice_coords[1]]

def upsample_vector(a, M: np.ndarray, original_shape, original_offset):
    Minv = np.linalg.inv(M)
    #original_shape = (7, 7)
    # This not only needs to create lattice_coords like original_shape,
    # but also add borders introduced by convolution
    x1 = Minv @ np.array([original_offset[0], original_offset[1]])
    x2 = Minv @ np.array([original_offset[0] + original_shape[0] - 1, original_offset[1]])
    x3 = Minv @ np.array([original_offset[0], original_offset[1] + original_shape[1] - 1])
    x4 = Minv @ np.array([original_offset[0] + original_shape[0] - 1, original_offset[1] + original_shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((original_shape[0],original_shape[1]), dtype=np.float64), np.array([original_offset[0], original_offset[1]]))

    lattice_coords = np.mgrid[original_offset[0]:original_offset[0] + original_shape[0],
                          original_offset[1]:original_offset[1] + original_shape[1]].reshape(2, -1)
    Minv_pre = np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])
    ups_coords = Minv_pre @ lattice_coords
    m = round(np.abs(np.linalg.det(M)))
    mask = np.all(np.mod(ups_coords, m) == 0, axis=0)
    ups_coords = to_python_vect(lattice_coords.T[mask], original_offset)
    upsampled.matrix[ups_coords[0], ups_coords[1]] = a
    return upsampled
