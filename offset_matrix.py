import numpy as np

class OffsetTensor:
    tensor: np.ndarray
    offset: np.ndarray

    def __init__(self, matrix: np.ndarray, offset):
        self.tensor = matrix
        if matrix.ndim == 3:
            self.tensor = matrix[:, :, 0]
        self.offset = offset

    def __add__(self, other):

        near_size = [min(self.offset[i], other.offset[i]) for i in range(len(self.offset))]
        far_size = [max(self.offset[i] + self.tensor.shape[i] - 1, other.offset[i] + other.tensor.shape[i] - 1) for i in range(len(self.offset))]
        self_offsets = np.array(
                [(self.offset[i] - near_size[i], far_size[i] - (self.offset[i] + self.tensor.shape[i] - 1)) for i in range(len(self.offset))]
                )
        other_offsets = np.array(
                [(other.offset[i] - near_size[i], far_size[i] - (other.offset[i] + other.tensor.shape[i] - 1)) for i in range(len(other.offset))]
                )
        self_padded = np.pad(self.tensor, self_offsets)
        other_padded = np.pad(other.tensor, other_offsets)
        return OffsetTensor(self_padded + other_padded, near_size)

    def __mul__(self, other):
        matrix = self.tensor * other
        return OffsetTensor(matrix, self.offset)