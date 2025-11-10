import itertools
from math import ceil, floor

import numpy as np

from classic.wave import convolve
from offset_tensor import OffsetTensor
from utils import to_python_vect


def downsample_vector(a: OffsetTensor, M: np.ndarray):
    m = round(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)
    Minv_pre = np.rint((m * Minv)).astype(np.int32)
    m = round(np.abs(np.linalg.det(M)))
    slices = tuple(slice(o, o + s) for s, o in zip(a.tensor.shape, a.offset))
    lattice_coords = np.mgrid[slices].reshape(a.offset.shape[0], -1)
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = tuple(to_python_vect(lattice_coords.T[mask], a.offset))
    return a.tensor[*lattice_coords]

def downsample_fast(a: OffsetTensor, M: np.ndarray):
    m = round(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)
    Minv_pre = np.rint((m * Minv)).astype(np.int32)
    slices = tuple(slice(0, m) for _ in range(a.offset.shape[0]))
    core_coords = np.mgrid[slices].reshape(a.offset.shape[0], -1)
    downs_coords = (Minv_pre @ core_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    core_coords = np.array(tuple(to_python_vect(core_coords.T[mask], a.offset)))
    core_points = a.tensor.shape // np.full_like(a.tensor.shape, m)
    core_slices = tuple(slice(0, m*cp, m) for cp in core_points)
    base_coords = np.mgrid[core_slices].reshape(a.offset.shape[0], -1)
    #print(core_coords)
    #print(base_coords)
    lattice_coords = (base_coords[..., None] + core_coords[:, None, :]).reshape(a.offset.shape[0], -1)
    #print(lattice_coords)
    return a.tensor[*lattice_coords]


def upsample_vector(a, M: np.ndarray, original_shape, original_offset):

    upsampled = OffsetTensor(np.zeros(original_shape, dtype=np.float64), np.array(original_offset))

    slices = tuple(slice(o, o + s) for s, o in zip(original_shape, original_offset))

    lattice_coords = np.mgrid[slices].reshape(np.array(original_shape).shape[0], -1)
    m = round(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)
    Minv_pre = np.rint((m * Minv)).astype(np.int32)
    ups_coords = Minv_pre @ lattice_coords
    m = round(np.abs(np.linalg.det(M)))
    mask = np.all(np.mod(ups_coords, m) == 0, axis=0)
    ups_coords = tuple(to_python_vect(lattice_coords.T[mask], original_offset))
    upsampled.tensor[*ups_coords] = a
    return upsampled


def transition_vector(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = mask.conjugate()
    return downsample_vector(convolve(a, mask), M)


def subdivision_vector(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray, original_shape, original_offset):
    u = upsample_vector(a,M, original_shape, original_offset)
    c = convolve(u, mask)
    return c
