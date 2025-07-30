import numpy as np
import scipy

from offset_matrix import OffsetMatrix
from utils import OffsetMatrixConjugate
from classic.sample import downsample, upsample


def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_offset = np.array([a.offset[0] + b.offset[0], a.offset[1] + b.offset[1]])
    new_matrix = scipy.signal.convolve2d(a.matrix, b.matrix, mode = 'full', boundary = 'fill')
    return OffsetMatrix(new_matrix, new_offset)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample(convolve(a, mask), M)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    u = upsample(a,M)
    c = convolve(u, mask)
    return c#convolve(upsample(a, M), mask)
