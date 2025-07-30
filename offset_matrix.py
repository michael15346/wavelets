import numpy as np

class OffsetMatrix:
    matrix: np.ndarray
    offset: np.ndarray

    def __init__(self, matrix: np.ndarray, offset):
        self.matrix = matrix
        if matrix.ndim == 3:
            self.matrix = matrix[:, :, 0]
        self.offset = offset

    def __add__(self, other):

        near_size = [min(self.offset[i], other.offset[i]) for i in range(len(self.offset))]
        far_size = [max(self.offset[i] + self.matrix.shape[i] - 1, other.offset[i] + other.matrix.shape[i] - 1) for i in range(len(self.offset))]
        self_offsets = np.array(
                [(self.offset[i] - near_size[i], far_size[i] - (self.offset[i] + self.matrix.shape[i] - 1)) for i in range(len(self.offset))]
                )
        other_offsets = np.array(
                [(other.offset[i] - near_size[i], far_size[i] - (other.offset[i] + other.matrix.shape[i] - 1)) for i in range(len(other.offset))]
                )
        self_padded = np.pad(self.matrix, self_offsets)
        other_padded = np.pad(other.matrix, other_offsets)
        return OffsetMatrix(self_padded + other_padded, near_size)

    def __mul__(self, other):
        matrix = self.matrix * other
        return OffsetMatrix(matrix, self.offset)