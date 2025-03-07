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
    h: OffsetMatrix
    g: OffsetMatrix
    hdual: OffsetMatrix
    gdual: OffsetMatrix
    M: OffsetMatrix
    m: float


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
    for gdual in w.gdual:
        d.append(transition(a, gdual, w.M))
    a = transition(a, w.hdual, w.M)
    return (a, d)

 
def idwt(a: OffsetMatrix, d: tuple[OffsetMatrix, ...], w: Wavelet):
    ai = subdivision(a, w.h, w.M)

    for i in range(len(w.g)):
        ai += subdivision(d[i], w.g[i], w.M)
    ai.matrix *= w.m
    return ai


def downsample_corners(a: OffsetMatrix, coords: tuple[float, float, float, float], w: Wavelet):
    Minv = np.linalg.inv(w.M)
    x1 = Minv @ np.array([a.offset[0], a.offset[1]])
    x2 = Minv @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1]])
    x3 = Minv @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = Minv @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1] - a.matrix.shape[0]+1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    ymax = floor(max(x1[1], x2[1], x4[1], x4[1]))
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = coords
    scaled_dl = (Minv @ np.array([x1, y1]))
    scaled_dr = (Minv @ np.array([x2, y2]))
    scaled_ul = (Minv @ np.array([x3, y3]))
    scaled_ur = (Minv @ np.array([x4, y4]))
    dl = tuple(to_python(scaled_dl[0], scaled_dl[1], np.array([xmin, ymax])))
    dr = tuple(to_python(scaled_dr[0], scaled_dr[1], np.array([xmin, ymax])))
    ul = tuple(to_python(scaled_ul[0], scaled_ul[1], np.array([xmin, ymax])))
    ur = tuple(to_python(scaled_ur[0], scaled_ur[1], np.array([xmin, ymax])))
    return (dl, dr, ul, ur)


def upsample_corners(a: OffsetMatrix, coords: tuple[float, float, float, float], w: Wavelet):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[1]  - 1, a.offset[1] - a.matrix.shape[0] + 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = coords
    scaled_dl = (M @ np.array([x1, y1]))
    scaled_dr = (M @ np.array([x2, y2]))
    scaled_ul = (M @ np.array([x3, y3]))
    scaled_ur = (M @ np.array([x4, y4]))
    dl = tuple(to_python(scaled_dl[0], scaled_dl[1], np.array([xmin, ymax])))
    dr = tuple(to_python(scaled_dr[0], scaled_dr[1], np.array([xmin, ymax])))
    ul = tuple(to_python(scaled_ul[0], scaled_ul[1], np.array([xmin, ymax])))
    ur = tuple(to_python(scaled_ur[0], scaled_ur[1], np.array([xmin, ymax])))
    return (dl, dr, ul, ur)


def wavedec(a0: OffsetMatrix, rank: int, w: Wavelet):
    a = a0
    d = list()
    x0 = a0.offset[0]
    y0 = a0.offset[0]
    (m, n) = a0.matrix.shape
    corners = ((x0, y0), (x0, y0 + n), (x0 + m, y0), (x0 + m, y0 + n))
    for i in range(rank):
        corners = downsample_corners(a, corners, w)
        (a, di) = dwt(a, w)
        d.append(di)
    return (a, d, corners)


def waverec(a: OffsetMatrix, d: list[OffsetMatrix], w: Wavelet, corners: tuple[tuple[float, float], ...]):
    ai = a
    for di in reversed(d):
        corners = upsample_corners(ai, corners, w)
        ai = idwt(ai, di, w)
    return (ai, corners)

#data = OffsetMatrix(iio.imread('http://upload.wikimedia.org/wikipedia/commons/d/de/Wikipedia_Logo_1.0.png'), np.array([0,0]))
data = OffsetMatrix(255 * np.array([[1, 1], [1, 1]]), np.array([0,0]))

print(data)
M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25], [0.5], [0.25]]), np.array([0,1]))
g = (OffsetMatrix(np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]]), np.array([0,3])),)
hdual = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual = (OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,2])),)

hdual_conj = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual_conj = (OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,0])),)
w = Wavelet(h, g, hdual_conj, gdual_conj, M, np.abs(np.linalg.det(M)))

(ai, d, corners_dec) = wavedec(data, 2, w)

(a, corners) = waverec(ai, d, w, corners_dec)
print(a.matrix)
print(corners)
print(to_python(corners[1][0], corners[1][1], a.offset))

