import numpy as np

from classic.wave import convolve
from db import SetOfDigitsFinder
from offset_tensor import OffsetTensor
from periodic.wave import convolve_period
from utils import to_python_vect


def downsample_ezw(a: OffsetTensor, ezw_coords: np.ndarray):
    return a.tensor[*ezw_coords]

def upsample_ezw(a: OffsetTensor, ezw_coords: np.ndarray, original_shape, original_offset):
    upsampled = OffsetTensor(np.zeros(original_shape, dtype=np.float64), np.array(original_offset))
    upsampled.tensor[*ezw_coords] = a
    return upsampled


def transition_ezw(a: OffsetTensor, mask: OffsetTensor, ezw_coords: np.ndarray):
    mask = mask.conjugate()
    return downsample_ezw(convolve_period(a, mask), ezw_coords)


def subdivision_ezw(a: OffsetTensor, mask: OffsetTensor, ezw_coords: np.ndarray, original_shape, original_offset):
    u = upsample_ezw(a, ezw_coords, original_shape, original_offset)
    c = convolve_period(u, mask)
    return c

def init_coords_ezw(shape, offset, level, M):
    Mp = np.linalg.matrix_power(M, level)
    mp = round(np.abs(np.linalg.det(Mp)))
    Mpinv = np.linalg.inv(Mp)
    Mpinv_pre = np.rint(mp * Mpinv).astype(np.int32)
    slices = tuple(slice(o, o + s) for s, o in zip(shape, offset))
    lattice_coords = np.mgrid[slices].reshape(offset.shape[0], -1)
    downs_coords = (Mpinv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, mp) == 0, axis=0)
    lattice_coords = np.array(tuple(to_python_vect(lattice_coords.T[mask], offset)))
    return lattice_coords

def step_coords_ezw(shape, lattice_coords, Mdigits):


    raw_coords = (lattice_coords[..., None] + Mdigits[:, None, :]).reshape(lattice_coords.shape[0], -1)
    ezw_coords = np.mod(raw_coords, shape[..., None])

    return ezw_coords

def gen_coords_ezw(shape, offset, level, M):
    Mp = np.linalg.matrix_power(M, level)
    mp = round(np.abs(np.linalg.det(Mp)))
    Mpinv = np.linalg.inv(Mp)
    Mpinv_pre = np.rint(mp * Mpinv).astype(np.int32)
    slices = tuple(slice(o, o + s) for s, o in zip(shape, offset))
    lattice_coords = np.mgrid[slices].reshape(offset.shape[0], -1)
    downs_coords = (Mpinv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, mp) == 0, axis=0)
    lattice_coords = np.array(tuple(to_python_vect(lattice_coords.T[mask], offset)))
    digits = SetOfDigitsFinder(M)
    Mdigits = [M @ digits.T]
    for l in range(level - 2):
        Mdigits.append(M @ Mdigits[-1])
    ezw_coords = [lattice_coords]

    for l in range(level - 1):
        raw_coords = (ezw_coords[-1][..., None] + Mdigits[-l-1][:, None, :]).reshape(lattice_coords.shape[0], -1)
        ezw_coords.append(np.mod(raw_coords, shape[..., None]))
    ezw_coords.reverse()
    return ezw_coords