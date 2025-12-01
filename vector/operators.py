import itertools
from math import ceil, floor

import numpy as np

from classic.wave import convolve
from offset_tensor import OffsetTensor
from utils import to_python_vect


def get_adjugate(M):
    Minv = np.linalg.inv(M)
    m = np.abs(np.linalg.det(M))
    Madj = np.rint((m * Minv)).astype(int)
    return Madj


# получение размеров фундаментальной плитки
def get_tile(M):
    Madj = get_adjugate(M)
    m = np.rint(np.abs(np.linalg.det(M))).astype(int)
    # получение знаменателей в несократимых дробях Madj / m.
    pre_tile = m // np.gcd(Madj, m)
    # НОК от 1D-массива через np.lcm.reduce
    tile = np.array([np.lcm.reduce(vec) for vec in pre_tile.T])
    return tile


def get_pad_up_to(shape, M, level):
    Mlvl = np.linalg.matrix_power(M, level)
    tile = get_tile(Mlvl)
    pad_up_to = (np.ceil(shape / tile) * tile).astype(int)
    return pad_up_to


def get_M_multiples_coords(shape, offset, M):
    Madj = get_adjugate(M)
    m = np.rint(np.abs(np.linalg.det(M))).astype(int)
    tile = get_tile(M)
    d = offset.shape[0]
    slices = tuple(slice(0, tile[i]) for i in range(d))
    in_tile_coords = np.mgrid[slices].reshape(d, -1)

    mask = np.all(np.mod(Madj @ in_tile_coords, m) == 0, axis=0)
    M_multiples_in_tile = np.array(tuple(to_python_vect(in_tile_coords.T[mask], offset)))
    tile_shifts = shape // tile
    shifts_slices = tuple(slice(0, tile[idx] * ts, tile[idx]) for idx, ts in enumerate(tile_shifts))
    shifts = np.mgrid[shifts_slices].reshape(d, -1)

    M_multiples_coords = (shifts[..., None] + M_multiples_in_tile[:, None, :]).reshape(d, -1)
    return M_multiples_coords


def downsample_fast(a: OffsetTensor, M: np.ndarray):
    M_multiples_coords = get_M_multiples_coords(a.tensor.shape, a.offset, M)
    return a.tensor[*M_multiples_coords]


def upsample_fast(a, M: np.ndarray, original_shape, original_offset):
    upsampled = OffsetTensor(np.zeros(original_shape, dtype=np.float64), np.array(original_offset))
    M_multiples_coords = get_M_multiples_coords(original_shape, original_offset, M)
    upsampled.tensor[*M_multiples_coords] = a
    return upsampled


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
