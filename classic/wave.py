from itertools import product

import numpy as np
import scipy

from offset_tensor import OffsetTensor
from utils import to_python_vect

def convolve(a: OffsetTensor, b: OffsetTensor):
    new_offset = a.offset + b.offset
    new_matrix = scipy.signal.oaconvolve(a.tensor, b.tensor, mode ='full')
    return OffsetTensor(new_matrix, new_offset)

def subdivision(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    u = upsample(a,M)
    c = convolve(u, mask)
    return c

def upsample(a: OffsetTensor, M: np.ndarray):
    choices = [(a.offset[i], a.offset[i] + a.tensor.shape[i] - 1) for i in range(len(a.tensor.shape))]
    corners = list(product(*choices))
    xs = M @ np.array(corners).T
    minc = np.ceil(np.min(xs, axis=1))
    maxc = np.floor(np.max(xs, axis=1))
    upsampled = OffsetTensor(np.zeros(maxc - minc + 1, dtype=np.float64), minc)
    slices = tuple(slice(o, o + s) for s, o in zip(a.tensor.shape, a.offset))
    lattice_coords = np.mgrid[slices].reshape(len(a.tensor.shape), -1)

    ups_coords = M @ lattice_coords
    lattice_coords = tuple(to_python_vect(lattice_coords.T, a.offset))
    ups_coords = tuple(to_python_vect(ups_coords.T, upsampled.offset))
    upsampled.tensor[*ups_coords] = a.tensor[*lattice_coords]
    return upsampled
