import numpy as np

from classic.operators import convolve
from offset_matrix import OffsetMatrix
from utils import OffsetMatrixConjugate
from vector.sample import downsample_vector, upsample_vector


def transition_vector(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample_vector(convolve(a, mask), M)


def subdivision_vector(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray, original_shape, original_offset):
    u = upsample_vector(a,M, original_shape, original_offset)
    c = convolve(u, mask)
    return c#convolve(upsample(a, M), mask)
