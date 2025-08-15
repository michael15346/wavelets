import itertools
from math import ceil, floor

import numpy as np

from classic.wave import convolve
from offset_matrix import OffsetTensor
from utils import OffsetMatrixConjugate, to_python_vect


def downsample_vector(a: OffsetTensor, M: np.ndarray):
    Minv_pre = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
    m = round(np.abs(np.linalg.det(M)))
    slices = tuple(slice(o, o + s) for s, o in zip(a.tensor.shape, a.offset))
    lattice_coords = np.mgrid[slices].reshape(a.offset.shape[0], -1)
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)
    return a.tensor[lattice_coords]

def upsample_vector(a, M: np.ndarray, original_shape, original_offset):
    # This not only needs to create lattice_coords like original_shape,
    # but also add borders introduced by convolution

    upsampled = OffsetTensor(np.zeros(original_shape, dtype=np.float64), np.array(original_offset))

    slices = tuple(slice(o, o + s) for s, o in zip(original_shape, original_offset))

    lattice_coords = np.mgrid[slices].reshape(original_shape.shape[0], -1)
    Minv_pre = np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])
    ups_coords = Minv_pre @ lattice_coords
    m = round(np.abs(np.linalg.det(M)))
    mask = np.all(np.mod(ups_coords, m) == 0, axis=0)
    ups_coords = to_python_vect(lattice_coords.T[mask], original_offset)
    upsampled.tensor[ups_coords] = a
    return upsampled


def transition_vector(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample_vector(convolve(a, mask), M)


def subdivision_vector(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray, original_shape, original_offset):
    u = upsample_vector(a,M, original_shape, original_offset)
    c = convolve(u, mask)
    return c
