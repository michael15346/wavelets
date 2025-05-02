import numpy as np
import scipy.signal
from skimage.metrics import structural_similarity as ssim
import imageio.v3 as iio
from dataclasses import dataclass
from math import ceil, floor
import kmeans1d
import line_profiler


class OffsetMatrix:
    matrix: np.ndarray
    offset: tuple

    def __init__(self, matrix: np.ndarray, offset):
        self.matrix = matrix
        if (matrix.ndim == 3):
            self.matrix = matrix[:, :, 0]
        self.offset = offset

    @line_profiler.profile
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
    g: tuple[OffsetMatrix, ...]
    hdual: OffsetMatrix
    gdual: tuple[OffsetMatrix, ...]
    M: np.ndarray
    m: float

    def __init__(self, h, g, hdual, gdual, M, m):
        self.h = h
        self.g = g
        self.hdual = hdual
        self.gdual = gdual
        self.M = M
        self.m = m


def to_python(x, y, offset):
    return offset[1]-y, x-offset[0]


def to_coord(row, col, offset):
    return offset[0]+col, offset[1]-row


def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_offset = (a.offset[0] + b.offset[0], a.offset[1] + b.offset[1])
    new_matrix = scipy.signal.convolve2d(a.matrix, b.matrix, mode = 'full', boundary = 'fill')
    return OffsetMatrix(new_matrix, new_offset)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):

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


def wavedec(a0: OffsetMatrix, rank: int, w: Wavelet):
    a = a0
    d = list()
    (m, n) = a0.matrix.shape
    for i in range(rank):
        a, di = dwt(a, w)
        d.append(di)
    return (a, d)


def waverec(a: OffsetMatrix, d: list[tuple[OffsetMatrix, ...]], w: Wavelet, original_shape: tuple[int, ...]) -> np.ndarray:
    ai = a
    for i, di in enumerate(reversed(d)):
        ai = idwt(ai, di, w)
        iio.imwrite(f'a{i}.png', np.clip(ai.matrix, 0, 255).astype(np.uint8))
    x1 = ai.offset[1]
    y1 = -ai.offset[0]
    x2 = x1 + original_shape[0]
    y2 = y1 + original_shape[1]
    ai = ai.matrix[x1:x2, y1:y2]
    return ai


def clamp(a: OffsetMatrix, d: OffsetMatrix):
    clamped = np.sum(np.abs(ai.matrix) <= 10)
    ai.matrix = np.where(np.abs(ai.matrix) > 10, ai.matrix, 0)
    total = ai.matrix.size
    for di in d:
        for dj in di:
            total += dj.matrix.size
            clamped += np.sum(np.abs(dj.matrix) <= 10)
            dj.matrix = np.where(np.abs(dj.matrix) > 10, dj.matrix, 0)
    print("clamp stats: total ", total, " clamped ", clamped, " clamped/total ", clamped / total)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    a_flat = a.flatten()
    b_flat = b.flatten()
    se = 0.
    for i in range(a_flat.size):
        se += (a_flat[i] - b_flat[i]) ** 2

    return np.sqrt(se / a_flat.size)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    a_flat = a.flatten()
    return 20 * np.log10(np.max(a_flat) / rmse(a, b))

# UPDATE!!!
def OffsetMatrix2CoefCoords(a: OffsetMatrix):
    shape = a.matrix.shape
    offset = a.offset
    coef = []
    coords = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if a.matrix[i, j] != 0:
                coef.append(a.matrix[i, j])
                coords.append([offset[0]+j, offset[1]-i])
    if len(coef) == 0:
        coef, coords = [0], [[0,0]]
    return (np.array(coef), np.array(coords))

def CoefCoords2OffsetMatrix(a):  # a filter in coef-coords form
    coef = a[0]
    coords = a[1]
    x_left, y_bottom = list(np.min(np.transpose(coords), axis=1))
    x_right, y_up = list(np.max(np.transpose(coords), axis=1))
    shape = [y_up-y_bottom+1, x_right-x_left+1]
    offset = np.array([x_left, y_up])
    matrix = np.zeros(shape)
    for ind, coord in enumerate(coords):
        row, col = to_python(coord[0], coord[1], offset)
        matrix[row, col] = coef[ind]
    return OffsetMatrix(matrix, offset)


def OffsetMatrixConjugate(a: OffsetMatrix):
    coef, coords = OffsetMatrix2CoefCoords(a)
    coords = - coords
    a_conj = CoefCoords2OffsetMatrix((coef, coords))
    return a_conj

def to_python_vect(coords, offset):
    # all_x, all_y
    return offset[1]-coords.T[1], coords.T[0]-offset[0]


def downsample(a: OffsetMatrix, M: np.ndarray):
    #print(M)
    Minv_pre = np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])
    m = int(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)
    x1 = Minv @ np.array([a.offset[0], a.offset[1]])
    x2 = Minv @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1]])
    x3 = Minv @ np.array([a.offset[0], a.offset[1] - a.matrix.shape[0]+1])
    x4 = Minv @ np.array([a.offset[0] + a.matrix.shape[1]-1, a.offset[1] - a.matrix.shape[0]+1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x4[1], x4[1]))

    downsampled = OffsetMatrix(np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.float64), np.array([xmin, ymax]))

    lattice_coords = np.mgrid[a.offset[0]:(a.offset[0] + a.matrix.shape[1]), (a.offset[1] - a.matrix.shape[0]+1):(a.offset[1]+1)].reshape(2, -1)  
    print(lattice_coords)
    downs_coords = Minv_pre @ lattice_coords
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)
    downs_coords = to_python_vect(downs_coords.T[mask]//m, downsampled.offset)

    downsampled.matrix[downs_coords[0], downs_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
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
    upsampled = OffsetMatrix(np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.float64), np.array([xmin, ymax]))
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0]+a.matrix.shape[1],
                              a.offset[1]-a.matrix.shape[0]+1:a.offset[1]+1]\
                       .reshape(2, -1)

    ups_coords = M @ lattice_coords
    lattice_coords = to_python_vect(lattice_coords.T, a.offset)
    ups_coords = to_python_vect(ups_coords.T, upsampled.offset)
    upsampled.matrix[ups_coords[0], ups_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
    return upsampled


def wavedec_multilevel_at_once(a: OffsetMatrix, w: Wavelet, level: int):
    masks = [list(map(OffsetMatrixConjugate, w.gdual))]
    nmasks = len(w.gdual)

    for i in range(1, level):
        gmasks = []
        for g in w.gdual:
            cur_mask = w.hdual
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask = convolve(upsample(w.hdual, cur_M), cur_mask)
                cur_M @= w.M
            cur_mask = convolve(upsample(g, cur_M), cur_mask)
            gmasks.append(cur_mask)
        masks.append(gmasks)
    
    cur_mask = w.hdual
    other_M = w.M.copy()
    for j in range(level-1, 0, -1):
        cur_mask = convolve(upsample(w.hdual, other_M), cur_mask)
        other_M @= w.M
    masks.append([OffsetMatrixConjugate(cur_mask), OffsetMatrixConjugate(cur_mask)])


    details = []
    cur_M = w.M.copy()
    for m in masks:
        #print(cur_M)
        details.append(list(map(
            transition, [a] * len(m), m, [cur_M] * len(m))))
        print("cur m  * len m: ", [cur_M] * len(m))
        print("cur m: ", cur_M)
        cur_M @= w.M

    return details









data = OffsetMatrix(iio.imread('test/lenna.bmp'), np.array([0,0]))
#data = OffsetMatrix(255 * np.array([[1, 1], [1, 1]]), np.array([0,0]))

print(data)
M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25], [0.5], [0.25]]), np.array([0,1]))
g = (OffsetMatrix(np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]]), np.array([0,3])),)
hdual = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual = (OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,2])),)

hdual_conj = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual_conj = (OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,0])),)


w = Wavelet(h, g, hdual, gdual, M, np.abs(np.linalg.det(M)))

#ai, d = wavedec(data, 5, w)
details = wavedec_multilevel_at_once(data, w, 2)
print(details)
for i, d in enumerate(details):
    for j, dd in enumerate(d):
        iio.imwrite(f'd{i}-{j}.png', np.clip(dd.matrix* (w.m ** (i +2)), 0, 255).astype(np.uint8))
#print(ai)
#for dd in d:
#    for ddd in dd:
#        print(ddd.matrix)
#
#print()
#
#iio.imwrite('a.png', np.clip(ai.matrix, 0, 255).astype(np.uint8))
#for i, di in enumerate(d):
#    for j, dij in enumerate(di):
#        iio.imwrite(f'd{i}-{j}.png', np.clip(dij.matrix * (w.m ** (5 -i )), 0, 255).astype(np.uint8))
##clamp(ai, d)
#
#a = waverec(ai, d, w, data.matrix.shape)
#print(a)
#print(np.clip(a, 0, 255).astype(np.uint8))
#print("RMSE:", rmse(data.matrix, np.clip(a, 0, 255)))
#print("PSNR:", psnr(data.matrix, np.clip(a, 0, 255)))
#print("SSIM:", ssim(data.matrix, np.clip(a, 0, 255), data_range=255))
#iio.imwrite('restored.png', np.clip(a, 0, 255).astype(np.uint8))
