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
    new_offset = np.array([a.offset[0] + b.offset[0], a.offset[1] + b.offset[1]])
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

    x1 = Minv @ np.array([a.offset[0], a.offset[1]])
    x2 = Minv @ np.array([a.offset[0] + a.tensor.shape[0] - 1, a.offset[1]])
    x3 = Minv @ np.array([a.offset[0], a.offset[1] + a.tensor.shape[1] - 1])
    x4 = Minv @ np.array([a.offset[0] + a.tensor.shape[0] - 1, a.offset[1] + a.tensor.shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))

    downsampled = OffsetTensor(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))



    lattice_coords = np.mgrid[a.offset[0]:a.offset[0] + a.tensor.shape[0],
                     a.offset[1]:a.offset[1] + a.tensor.shape[1]].reshape(2, -1)
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)

    downs_coords = list(to_python_vect(downs_coords.T[mask ]//m, downsampled.offset))
    downsampled.tensor[downs_coords[0], downs_coords[1]] = a.tensor[lattice_coords[0], lattice_coords[1]]
    return downsampled


def upsample(a: OffsetTensor, M: np.ndarray):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.tensor.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] + a.tensor.shape[1] - 1])
    x4 = M @ np.array([a.offset[0] + a.tensor.shape[0] - 1, a.offset[1] + a.tensor.shape[1] - 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetTensor(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0]+a.tensor.shape[0],
                              a.offset[1]:a.offset[1]+a.tensor.shape[1]]\
                       .reshape(2, -1)

    ups_coords = M @ lattice_coords
    lattice_coords = to_python_vect(lattice_coords.T, a.offset)
    ups_coords = to_python_vect(ups_coords.T, upsampled.offset)
    upsampled.tensor[ups_coords[0], ups_coords[1]] = a.tensor[lattice_coords[0], lattice_coords[1]]
    return upsampled


def wavedec(a0: OffsetTensor, rank: int, w: Wavelet):
    a = a0
    d = list()
    for i in range(rank):
        a, di = dwt(a, w)
        d.append(di)
    d.reverse()
    c = [a] + d
    return c


def waverec(c: list, w: Wavelet, original_shape: tuple[int, ...]) -> np.ndarray:
    ai = c[0]
    d = c[1:]
    for i, di in enumerate(d):
        ai = idwt(ai, di, w)
    x1 = -ai.offset[0]
    y1 = -ai.offset[1]
    x2 = x1 + original_shape[0]
    y2 = y1 + original_shape[1]
    ai = ai.tensor[x1:x2, y1:y2]
    return ai
