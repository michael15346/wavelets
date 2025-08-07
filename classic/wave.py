import numpy as np

from offset_matrix import OffsetMatrix
from wavelet import Wavelet
from classic.transform import dwt, idwt


def wavedec(a0: OffsetMatrix, rank: int, w: Wavelet):
    a = a0
    d = list()
    (m, n) = a0.matrix.shape
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
    ai = ai.matrix[x1:x2, y1:y2]
    return ai
