from itertools import product
from math import floor, ceil

import numpy as np
import scipy

from offset_matrix import OffsetTensor
from utils import OffsetMatrixConjugate, to_python_vect
from wavelet import Wavelet

def dwt(a: OffsetTensor, w: Wavelet):
    d = list()
    for gdual in w.gdual:
        d.append(transition(a, gdual, w.M))#
    a = transition(a, w.hdual, w.M)
    return a, d


def idwt(a: OffsetTensor, d: tuple[OffsetTensor, ...], w: Wavelet):
    ai = subdivision(a, w.h, w.M)

    for i in range(len(w.g)):
        ai += subdivision(d[i], w.g[i], w.M)
    ai.tensor *= w.m
    return ai


def convolve(a: OffsetTensor, b: OffsetTensor):
    new_offset = a.offset + b.offset
    new_matrix = scipy.signal.convolve2d(a.tensor, b.tensor, mode ='full', boundary ='fill')
    return OffsetTensor(new_matrix, new_offset)


def transition(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample(convolve(a, mask), M)


def subdivision(a: OffsetTensor, mask: OffsetTensor, M: np.ndarray):
    u = upsample(a,M)
    c = convolve(u, mask)
    return c


def downsample(a: OffsetTensor, M: np.ndarray):
    Minv_pre = np.array([[M[1 ,1] ,-M[0 ,1]] ,[-M[1 ,0] ,M[0 ,0]]])
    m = round(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)
    choices = [(a.offset[i], a.offset[i] + a.tensor.shape[i] - 1) for i in range(len(a.tensor.shape))]
    corners = list(product(*choices))
    xs = Minv @ np.array(corners).T
    minc = np.array(np.ceil(np.min(xs, axis=1)), dtype=int)
    maxc = np.array(np.floor(np.max(xs, axis=1)), dtype=int)
    downsampled = OffsetTensor(np.zeros(maxc - minc + 1, dtype=np.float64), minc)
    slices = tuple(slice(o, o + s) for s, o in zip(a.tensor.shape, a.offset))
    lattice_coords = np.mgrid[slices].reshape(len(a.tensor.shape), -1)
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = tuple(to_python_vect(lattice_coords.T[mask], a.offset))

    downs_coords = tuple(to_python_vect(downs_coords.T[mask ]//m, downsampled.offset))
    downsampled.tensor[*downs_coords] = a.tensor[*lattice_coords]
    return downsampled


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


def wavedec(a0: OffsetTensor, w: Wavelet, rank: int):
    a = a0
    d = list()
    for i in range(rank):
        a, di = dwt(a, w)
        d.append(di)
    d.reverse()
    c = [a] + d
    return c


def waverec(c: list, w: Wavelet, original_shape: tuple[int, ...]) -> OffsetTensor:
    ai = c[0]
    d = c[1:]
    for i, di in enumerate(d):
        ai = idwt(ai, di, w)
    slices = tuple(slice(-o, -o + s) for s, o in zip(original_shape, ai.offset))
    ai = OffsetTensor(ai.tensor[slices], np.zeros_like(ai.offset))
    return ai
