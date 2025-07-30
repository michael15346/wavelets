from math import floor, ceil

import numpy as np

from offset_matrix import OffsetMatrix
from utils import to_python_vect


def downsample(a: OffsetMatrix, M: np.ndarray):
    Minv_pre = np.array([[M[1 ,1] ,-M[0 ,1]] ,[-M[1 ,0] ,M[0 ,0]]])
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

    downs_coords = list(to_python_vect(downs_coords.T[mask ]//m, downsampled.offset))
    downsampled.matrix[downs_coords[0], downs_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
    return downsampled


def upsample(a: OffsetMatrix, M: np.ndarray):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] + a.matrix.shape[1]-1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[0]  - 1, a.offset[1] + a.matrix.shape[1] - 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0]+a.matrix.shape[0],
                              a.offset[1]:a.offset[1]+a.matrix.shape[1]]\
                       .reshape(2, -1)

    ups_coords = M @ lattice_coords
    lattice_coords = to_python_vect(lattice_coords.T, a.offset)
    ups_coords = to_python_vect(ups_coords.T, upsampled.offset)
    upsampled.matrix[ups_coords[0], ups_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
    return upsampled
