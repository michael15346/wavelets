import numpy as np
import scipy.signal
import imageio.v3 as iio
from dataclasses import Dataclass


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


@Dataclass
class Wavelet:
    h: np.ndarray
    g: np.ndarray
    hdual: np.ndarray
    gdual: np.ndarray
    M: np.ndarray


def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_coords = (a.coords[0] + b.coords[0], a.coords[1] + b.coords[1])
    new_matrix = scipy.signal.convole(a.matrix, b.matrix, 'full')
    return OffsetMatrix(new_matrix, new_coords)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    Minv = np.linalg.inv(M)
    return downscale(convolve(a, mask), Minv)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    return convolve(upscale(a, M), mask)


def dwt(a: OffsetMatrix, w: Wavelet):
    a0 = subdivision(a, w.h, w.M)
    d0 = subdivision(a, w.g, w.M)
    return (a0, d0)


def idwt(a0: OffsetMatrix, d: OffsetMatrix, w: Wavelet):
    ai = transition(a0, w.h, w.M)
    di = transition(d, w.g, w.M)
    return (ai + di)
#TODO: сохранять координаты параллелограмма и по ним обрезать итоговое изображение
#data = iio.imread('http://upload.wikimedia.org/wikipedia/commons/d/de/Wikipedia_Logo_1.0.png')
data = 255 * np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

print(data)
M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25], [0.5], [0.25]]), )
g = np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]])
hdual = np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]])
gdual = np.array([[-0.25], [0.5], [-0.25]])

ai, d = dwt(data, h, g, M, 5)
print(ai, d)
a = idwt(ai, d, hdual, gdual, M)
iio.imwrite('image.png', a.astype(np.uint8))

