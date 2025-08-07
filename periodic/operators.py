import numpy as np
import scipy

from classic.operators import convolve
from offset_matrix import OffsetMatrix
from utils import OffsetMatrixConjugate
from vector.sample import downsample_vector, upsample_vector


def transition_period(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample_vector(convolve_period(a, mask), M)


def subdivision_period(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray, original_shape, original_offset):
    u = upsample_vector(a,M, original_shape, original_offset)
    c = convolve_period(u, mask)
    return c#convolve(upsample(a, M), mask)

def convolve_period(a: OffsetMatrix, b: OffsetMatrix):
    new_offset = a.offset + b.offset + np.ceil(np.array(b.matrix.shape)/2) -1
    # !!! only 2D
    new_tensor = scipy.signal.convolve2d(a.matrix, b.matrix, mode = 'same', boundary = 'wrap')
    # !!! may be faster, but computation errors
    #new_tensor = scipy.signal.oaconvolve(a.tensor, b.tensor, mode = 'full')
    # !!! a bit slower
    #new_tensor = scipy.signal.convolve(a.tensor, b.tensor, mode = 'full', method = 'direct')
    #print(new_offset)
    # периодический сдвиг, чтобы (0,0) координата была в индексе (0,0)
    new_tensor = np.roll(new_tensor, tuple(new_offset.astype(np.int32)), axis=(0, 1))
    #return OffsetTensor(new_tensor, new_offset)
    return OffsetMatrix(new_tensor, np.array([0,0]))