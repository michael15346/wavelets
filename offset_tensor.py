from copy import deepcopy

import numpy as np

class OffsetTensor:
    tensor: np.ndarray
    offset: np.ndarray

    def __init__(self, matrix: np.ndarray, offset):
        self.tensor = matrix
        self.offset = np.array(offset)

    def __add__(self, other):

        near_size = np.min(np.stack([self.offset, other.offset]), axis = 0)
        far_size = np.max(np.stack([self.offset + self.tensor.shape, other.offset + other.tensor.shape]), axis = 0)
        self_offsets = np.column_stack(
                (self.offset - near_size, far_size - (self.offset + self.tensor.shape))
                ) # как zip() для numpy
        other_offsets = np.column_stack(
                (other.offset - near_size, far_size - (other.offset + other.tensor.shape))
                )
        self_padded = np.pad(self.tensor, self_offsets)
        other_padded = np.pad(other.tensor, other_offsets)
        return OffsetTensor(self_padded + other_padded, near_size)

    def __mul__(self, other):
        matrix = self.tensor * other
        return OffsetTensor(matrix, self.offset)

    def conjugate(self):
        a_conj = deepcopy(self)
        np.flip(a_conj.tensor)
        a_conj.offset = -(self.offset + self.tensor.shape - 1)
        return a_conj