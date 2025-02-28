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
        #print(self.matrix)
        print("other shape:", other.matrix.shape)
        #print(other.matrix)

        self_left = max(self.offset[0] - other.offset[0], 0)
        other_left = max(other.offset[0] - self.offset[0], 0)

        self_top = max(other.offset[1] - self.offset[1], 0)
        other_top = max(self.offset[1] - other.offset[1], 0)
        self_right = max((other.matrix.shape[1] + other.offset[0]) -
                         (self.matrix.shape[1] + self.offset[0]), 0)
        other_right = max((self.matrix.shape[1] + self.offset[0]) -
                          (other.matrix.shape[1] + other.offset[0]), 0)
        self_bottom = max((self.offset[1] - self.matrix.shape[0]) -
                          (other.offset[1] - other.matrix.shape[0]), 0)
        other_bottom = max((other.offset[1] - other.matrix.shape[0]) -
                           (self.offset[1] - self.matrix.shape[0]), 0)
        print("self offsets: l t r b", self_left, self_top, self_right, self_bottom)
        print("other offsets: l t r b", other_left, other_top, other_right, other_bottom)
        self_padded = np.pad(self.matrix, ((self_top, self_bottom), (self_left, self_right)))
        other_padded = np.pad(other.matrix, ((other_top, other_bottom),(other_left, other_right)))
        return OffsetMatrix(self_padded + other_padded, (self.offset[0] - self_left, self.offset[1] + self_top))


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

    Minv = np.linalg.inv(M)
    x1 = Minv @ np.array([a.offset[0], a.offset[1]])
    x2 = Minv @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1]])
    x3 = Minv @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = Minv @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1] - a.matrix.shape[0]+1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x4[1], x4[1]))

    downsampled = OffsetMatrix(np.zeros((ymax - ymin + 1, xmax - xmin + 1)), np.array([xmin, ymax]))
    print(downsampled.offset)
    for x in range(a.offset[0], a.offset[0] + a.matrix.shape[1]): 
        for y in range(a.offset[1], a.offset[1] - a.matrix.shape[0], -1):

            scaled = Minv @ np.array([x, y])
            if np.linalg.norm(scaled-scaled.astype(int)) < 0.0001:
                scaled = scaled.astype(int)
                downsampled.matrix[to_python(scaled[0], scaled[1], downsampled.offset)] = a.matrix[to_python(x, y, a.offset)]
    return downsampled

def upsample(a: OffsetMatrix, M: np.ndarray):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[1]  - 1, a.offset[1] - a.matrix.shape[0] + 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((ymax - ymin + 1, xmax - xmin + 1)), np.array([xmin, ymax]))
    for x in range(a.offset[0], a.offset[0] + a.matrix.shape[1]): 
        for y in range(a.offset[1], a.offset[1] - a.matrix.shape[0], -1):
            scaled = (M @ np.array([x, y])).astype(int)
            upsampled.matrix[to_python(scaled[0], scaled[1], upsampled.offset)] = a.matrix[to_python(x, y, a.offset)]
    return upsampled

def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_offset = (a.offset[0] - (b.offset[0] + b.matrix.shape[1] - 1), a.offset[1] - (b.offset[1] - (b.matrix.shape[0] - 1)))
    new_matrix = scipy.signal.convolve(a.matrix, b.matrix, 'full')
    return OffsetMatrix(new_matrix, new_offset)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):

    #здесь считать сопряженный фильтр. дуальные фильтры флипаются
    return downsample(convolve(a, mask), M)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    return convolve(upsample(a, M), mask)


def dwt(a: OffsetMatrix, w: Wavelet):
    d = list()
    for i in range(5):
        d.append(transition(a, w.gdual, w.M))
        iio.imwrite(f'd{i}.png', d[-1].matrix.astype(np.uint8))
        a = transition(a, w.hdual, w.M)
    return (a, d)

 
def idwt(a0: OffsetMatrix, d: list[OffsetMatrix], w: Wavelet):
    ai = a0
    i = 0
    m = np.linalg.det(w.M)
    for di in reversed(d):
        ai = subdivision(ai, w.h, w.M)
        dd = subdivision(di, w.g, w.M)
        ai = (ai + dd)
        ai.matrix *= m
        print(ai.matrix)
        print(dd.matrix)
        iio.imwrite(f'a{i}.png', ai.matrix.astype(np.uint8))
        i += 1
    print(ai.matrix)
    print(di.matrix)
    return ai

data = OffsetMatrix(iio.imread('http://upload.wikimedia.org/wikipedia/commons/d/de/Wikipedia_Logo_1.0.png'), np.array([0,0]))
#data = OffsetMatrix(255 * np.array([[1, 1], [1, 1]]), np.array([0,0]))

print(data)
M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25], [0.5], [0.25]]), np.array([0,1]))
g = OffsetMatrix(np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]]), np.array([0,3]))
hdual = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual = OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,2]))

hdual_conj = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual_conj = OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,0]))
#подумать над тем, как флипать маски, чтобы все координаты оставались питоновскими и все работало
w = Wavelet(h, g, hdual_conj, gdual_conj, M)

ai, d = dwt(data, w)
print(ai.matrix)
#print(d.matrix)
iio.imwrite('a.png', ai.matrix.astype(np.uint8))
iio.imwrite('d.png', ai.matrix.astype(np.uint8))
a = idwt(ai, d, w)
print(a.matrix)
iio.imwrite('image.png', a.matrix.astype(np.uint8))

