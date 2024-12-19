import numpy as np
import scipy.signal
import imageio.v3 as iio
from dataclasses import dataclass


class OffsetMatrix:
    matrix: np.ndarray
    coords: tuple

    def __init__(self, matrix, coords):
        self.matrix = matrix
        self.coords = coords

    def __add__(self, other):
        left_coords = (max(other.coords[0] - self.coords[0], 0), max(other.coords[1] - self.coords[1], 1))
        right_coords = (max(self.coords[0] - other.coords[0], 0), max(self.coords[1] - other.coords[1], 1))
        left_padded = np.pad(self.matrix, left_coords)
        right_padded = np.pad(other.matrix, right_coords)
        return OffsetMatrix(left_padded + right_padded, (0, 0))


@dataclass
class Wavelet:
    h: np.ndarray
    g: np.ndarray
    hdual: np.ndarray
    gdual: np.ndarray
    M: np.ndarray


# 
def downsample(a: OffsetMatrix, M: np.ndarray):

    x1 = M @ np.array([a.coords[0], a.coords[1]])
    x2 = M @ np.array([a.coords[0] + a.matrix.shape[0], a.coords[1]])
    x3 = M @ np.array([a.coords[0] , a.coords[1] + a.matrix.shape[1]])
    x4 = M @ np.array([a.coords[0] + a.matrix.shape[0], a.coords[1] + a.matrix.shape[1]])
    xmin = min(x1[0], x2[0], x3[0], x4[0])
    xmax = max(x1[0], x2[0], x3[0], x4[0])
    ymin = min(x1[1], x2[1], x3[1], x4[1])
    ymax = max(x1[1], x2[1], x4[1], x4[1]) ## try mesh grid, and use integer maths
                                            # try matrix idea (get matrix by all indices and then filter non-needed)

    ares = np.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
    for i in range(a.matrix.shape[0]): # range is wrong
        for j in range(a.matrix.shape[1]):
            x = M @ np.array([i, j])
            ares[int(x[0] - xmin), int(x[1] - ymin)] = a.matrix[i, j]
    base = M @ a.coords
    return OffsetMatrix(ares, base)
    
def upsample(a: OffsetMatrix, M: np.ndarray):
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    x1 = M @ np.array([a.coords[0], a.coords[1]])
    x2 = M @ np.array([a.coords[0] + a.matrix.shape[0], a.coords[1]])
    x3 = M @ np.array([a.coords[0] , a.coords[1] + a.matrix.shape[1]])
    x4 = M @ np.array([a.coords[0] + a.matrix.shape[0], a.coords[1] + a.matrix.shape[1]])
    xmin = min(x1[0], x2[0], x3[0], x4[0])
    xmax = max(x1[0], x2[0], x3[0], x4[0])
    ymin = min(x1[1], x2[1], x3[1], x4[1])
    ymax = max(x1[1], x2[1], x4[1], x4[1]) ## try mesh grid, and use integer maths
    ares = np.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
    for i in range(a.matrix.shape[0]): # range is wrong
        for j in range(a.matrix.shape[1]):
            x = M @ np.array([i, j])
            ares[int(x[0] - xmin), int(x[1] - ymin)] = a.matrix[i, j]
    base = M @ a.coords
    return OffsetMatrix(ares, base)

def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_coords = (a.coords[0] + b.coords[0], a.coords[1] + b.coords[1]) # на самом деле нужно вычитать правый нижний угол B
    new_matrix = scipy.signal.convolve(a.matrix, b.matrix, 'full')
    return OffsetMatrix(new_matrix, new_coords)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    Minv = np.linalg.inv(M)
    return downsample(convolve(a, mask), Minv)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    return convolve(upsample(a, M), mask)


def dwt(a: OffsetMatrix, w: Wavelet):
    a0 = subdivision(a, w.h, w.M)
    d0 = subdivision(a, w.g, w.M)
    return (a0, d0)

 
def idwt(a0: OffsetMatrix, d: OffsetMatrix, w: Wavelet):
    ai = transition(a0, w.hdual, w.M)
    di = transition(d, w.gdual, w.M)
    return (ai + di)
#TODO: сохранять координаты параллелограмма и по ним обрезать итоговое изображение
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
print(ai.matrix, d.matrix)
a = idwt(ai, d, w)
iio.imwrite('image.png', a.astype(np.uint8))

