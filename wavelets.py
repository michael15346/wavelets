import numpy as np
import scipy.signal
import imageio.v3 as iio
from dataclasses import dataclass
from math import ceil, floor



class OffsetMatrix:
    matrix: np.ndarray
    offset: tuple

    def __init__(self, matrix, offset):
        self.matrix = matrix
        self.offset = offset

    def __add__(self, other):
        print("offsets:", self.offset, other.offset)
        print("self shape:", self.matrix.shape)
        print(self.matrix)
        print("other shape:", other.matrix.shape)
        print(other.matrix)

        self_left = max(self.offset[0] - other.offset[0] + 3, 0)
        self_top = max(self.offset[1] - other.offset[1] - 1, 0)
        other_left = max(other.offset[0] - self.offset[0] - 3, 0) 
        other_top = max(other.offset[1] - self.offset[1] + 1, 0)
        self_right = max((other.matrix.shape[0] + other_left) -
                         (self.matrix.shape[0] + self_left), 0)
        other_right = max((self.matrix.shape[0] + self_left) -
                          (other.matrix.shape[0] + other_left), 0)
        self_bottom = max((other.matrix.shape[1] + other_top) -
                          (self.matrix.shape[1] + self_top), 0)
        other_bottom = max((self.matrix.shape[1] + self_top) -
                           (other.matrix.shape[1] + other_top), 0)
        print("self offsets: l t r b", self_left, self_top, self_right, self_bottom)
        print("other offsets: l t r b", other_left, other_top, other_right, other_bottom)
        self_padded = np.pad(self.matrix, ((self_left, self_right), (self_top, self_bottom)))
        other_padded = np.pad(other.matrix, ((other_left, other_right),(other_top, other_bottom)))
        return OffsetMatrix(self_padded + other_padded, (0, 0))


@dataclass
class Wavelet:
    h: np.ndarray
    g: np.ndarray
    hdual: np.ndarray
    gdual: np.ndarray
    M: np.ndarray


def to_python(x, y, offset):
    return offset[1]-y, x-offset[0]


def to_coord(row, col, offset):
    return offset[0]+col, offset[1]-row


def downsample(a: OffsetMatrix, M: np.ndarray):

    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1] - a.matrix.shape[0]+1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x4[1], x4[1]))

    downsampled = OffsetMatrix(np.zeros((ymax - ymin + 1, xmax - xmin + 1)), np.array([xmin, ymax]))
    print(downsampled.offset)
    for x in range(a.offset[0], a.offset[0] + a.matrix.shape[1]): 
        for y in range(a.offset[1], a.offset[1] - a.matrix.shape[0], -1):

            scaled = M @ np.array([x, y])
            if np.linalg.norm(scaled-scaled.astype(int)) < 0.0001:
                scaled = scaled.astype(int)
                downsampled.matrix[to_python(scaled[0], scaled[1], downsampled.offset)] = a.matrix[to_python(x, y, a.offset)]
    return downsampled

def upsample(a: OffsetMatrix, M: np.ndarray):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[1]  - 1, a.offset[1] - a.matrix.shape[0] + 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((ymax - ymin + 1, xmax - xmin + 1)), np.array([xmin, ymax]))
    for x in range(a.offset[0], a.offset[0] + a.matrix.shape[1]): 
        for y in range(a.offset[1], a.offset[1] - a.matrix.shape[0], -1):
            scaled = M @ np.array([x, y])
            if np.linalg.norm(scaled-scaled.astype(int)) < 0.0001:
                scaled = scaled.astype(int)
                upsampled.matrix[to_python(scaled[0], scaled[1], upsampled.offset)] = a.matrix[to_python(x, y, a.offset)]
    return upsampled

def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_offset = (a.offset[0] + b.offset[1] - b.matrix.shape[1] - 1, a.offset[1] + b.offset[0] - b.matrix.shape[0] - 1)
    new_matrix = scipy.signal.convolve(a.matrix, b.matrix, 'full')
    return OffsetMatrix(new_matrix, new_offset)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    return downsample(convolve(a, mask), M)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    Minv = np.linalg.inv(M)
    return convolve(upsample(a, Minv), mask)


def dwt(a: OffsetMatrix, w: Wavelet):
    a0 = transition(a, w.hdual, w.M)
    d0 = transition(a, w.gdual, w.M)
    return (a0, d0)

 
def idwt(a0: OffsetMatrix, d: OffsetMatrix, w: Wavelet):
    ai = subdivision(a0, w.h, w.M)
    di = subdivision(d, w.g, w.M)
    print(ai.matrix)
    print(di.matrix)
    return (ai + di)

#data = iio.imread('http://upload.wikimedia.org/wikipedia/commons/d/de/Wikipedia_Logo_1.0.png')
data = OffsetMatrix(255 * np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), np.array([0,0]))

print(data)
M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25], [0.5], [0.25]]), np.array([0,-1]))
g = OffsetMatrix(np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]]), np.array([0,-1]))
hdual = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,-2]))
gdual = OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,0]))

w = Wavelet(h, g, hdual, gdual, M)

ai, d = dwt(data, w)
print(ai.matrix)
print(d.matrix)
a = idwt(ai, d, w)
print(a.matrix)
iio.imwrite('image.png', a.matrix.astype(np.uint8))

