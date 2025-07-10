import numpy as np
import scipy.signal
from copy import deepcopy
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

        near_size = [min(self.offset[i], other.offset[i]) for i in range(len(self.offset))]
        far_size = [max(self.offset[i] + self.matrix.shape[i] - 1, other.offset[i] + other.matrix.shape[i] - 1) for i in range(len(self.offset))]
        self_offsets = tuple(
                [(self.offset[i] - near_size[i], far_size[i] - (self.offset[i] + self.matrix.shape[i] - 1)) for i in range(len(self.offset))]
                )
        other_offsets = tuple(
                [(other.offset[i] - near_size[i], far_size[i] - (other.offset[i] + other.matrix.shape[i] - 1)) for i in range(len(other.offset))]
                )
        self_padded = np.pad(self.matrix, self_offsets)
        other_padded = np.pad(other.matrix, other_offsets)
        return OffsetMatrix(self_padded + other_padded, near_size)

    def __mul__(self, other):
        matrix = self.matrix * other
        return OffsetMatrix(matrix, self.offset)

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
    return x - offset[0], y - offset[1]#offset[1]-y, x-offset[0]


def to_coord(row, col, offset):
    return offset[0]+col, offset[1]-row


def convolve(a: OffsetMatrix, b: OffsetMatrix):
    new_offset = (a.offset[0] + b.offset[0], a.offset[1] + b.offset[1])
    new_matrix = scipy.signal.convolve2d(a.matrix, b.matrix, mode = 'full', boundary = 'fill')
    return OffsetMatrix(new_matrix, new_offset)


def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample(convolve(a, mask), M)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    u = upsample(a,M)
    #print("u:",u.matrix)
    c = convolve(u, mask)
    #print("c:", c.matrix)
    return c#convolve(upsample(a, M), mask)


def dwt(a: OffsetMatrix, w: Wavelet):
    d = list()
    for gdual in w.gdual:
        d.append(transition(a, gdual, w.M))#
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
    d.reverse()
    c = [a] + d
    return c


def waverec(c: list, w: Wavelet, original_shape: tuple[int, ...]) -> np.ndarray:
    ai = c[0]
    d = c[1:]
    for i, di in enumerate(d):
        ai = idwt(ai, di, w)
    x1 = ai.offset[1]
    y1 = -ai.offset[0]
    x2 = x1 + original_shape[0]
    y2 = y1 + original_shape[1]
    ai = ai.matrix[x1:x2, y1:y2]
    return ai


def clamp(a: OffsetMatrix, d: tuple[OffsetMatrix, ...]):
    clamped = np.sum(np.abs(a.matrix) <= 10)
    a.matrix = np.where(np.abs(a.matrix) > 10, a.matrix, 0)
    total = a.matrix.size
    for di in d:
        for dj in di:
            total += dj.matrix.size
            clamped += np.sum(np.abs(dj.matrix) <= 10)
            dj.matrix = np.where(np.abs(dj.matrix) > 10, dj.matrix, 0)


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
                coords.append([offset[0]+i, offset[1]+j])
    if len(coef) == 0:
        coef, coords = [0], [[0,0]]
    return (np.array(coef), np.array(coords))

def CoefCoords2OffsetMatrix(a):  # a filter in coef-coords form
    coef = a[0]
    coords = a[1]
    x_left, y_bottom = list(np.min(np.transpose(coords), axis=1))
    x_right, y_up = list(np.max(np.transpose(coords), axis=1))
    shape = [x_right-x_left+1, y_up-y_bottom+1]
    offset = np.array([x_left, y_bottom])
    matrix = np.zeros(shape)
    for ind, coord in enumerate(coords):
        row, col = to_python(coord[0], coord[1], offset)
        matrix[row, col] = coef[ind]
    return OffsetMatrix(matrix, offset)


def OffsetMatrixConjugate(a: OffsetMatrix):
    a_conj = deepcopy(a)
    np.flip(a_conj.matrix)
    a_conj.offset = (-(a.offset[0] + a.matrix.shape[0] - 1), -(a.offset[1] + a.matrix.shape[1] - 1))


    return a_conj

def to_python_vect(coords, offset):
    # all_x, all_y
    #return offset[1]-coords.T[1], coords.T[0]-offset[0]
    return coords.T[0] - offset[0], coords.T[1]-offset[1]


def downsample(a: OffsetMatrix, M: np.ndarray):
    #print(a.matrix)
    Minv_pre = np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])
    m = round(np.abs(np.linalg.det(M)))
    Minv = np.linalg.inv(M)
    
    x1 = Minv @ np.array([a.offset[0], a.offset[1]])
    x2 = Minv @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = Minv @ np.array([a.offset[0], a.offset[1] + a.matrix.shape[1] - 1])
    x4 = Minv @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1] + a.matrix.shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))

    downsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))


    #print(list([a.offset[0]:(a.offset[0] + a.matrix.shape[0])]))
    
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0] + a.matrix.shape[0],
                              a.offset[1]:a.offset[1] + a.matrix.shape[1]].reshape(2, -1)  
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)
    
    downs_coords = list(to_python_vect(downs_coords.T[mask]//m, downsampled.offset))
    #print(downs_coords)
    #print(downs_coords)
    #print(lattice_coords)
    #print(a.matrix[lattice_coords[0], lattice_coords[1]])
    downsampled.matrix[downs_coords[0], downs_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
    return downsampled


def upsample(a: OffsetMatrix, M: np.ndarray):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] + a.matrix.shape[1]-1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[0]  - 1, a.offset[1] + a.matrix.shape[1] - 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymax]))
    #print(a.offset[0])
    #print(a.matrix.shape[0])
    #print(a.offset[1])
    #print(a.matrix.shape[1])
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0]+a.matrix.shape[0]-1,
                              a.offset[1]:a.offset[1]+a.matrix.shape[1]-1]\
                       .reshape(2, -1)

    ups_coords = M @ lattice_coords
    lattice_coords = to_python_vect(lattice_coords.T, a.offset)
    ups_coords = to_python_vect(ups_coords.T, upsampled.offset)
    #print("ups_coords:", ups_coords)
    upsampled.matrix[ups_coords[0], ups_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
    return upsampled


def wavedec_multilevel_at_once(data: OffsetMatrix, w: Wavelet, level: int):
    masks = [list(w.gdual)]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask = subdivision(w.hdual, cur_mask, cur_M)
                cur_M @= w.M
            wave_mask = subdivision(gdual, cur_mask, cur_M)
            gmasks.append(wave_mask)
        masks.append(gmasks)
    # !!!
    if level > 1:
        ref_mask = subdivision(w.hdual, cur_mask, cur_M)
    else:
        ref_mask = w.hdual
    #masks[-1].append(ref_mask)

    details = []
    cur_M = np.eye(w.M.shape[0], dtype=int)
    for mask in masks:
        cur_M @= w.M
        tmp_list = list()
        for m in mask:
            tmp_list.append(transition(data, m, cur_M.copy()))
        details.append(tmp_list)
        #details.append(list(map(
        #    transition, [data] * len(mask), mask, [cur_M] * len(mask))))
    details.append(transition(data, ref_mask, cur_M))
    details.reverse()

    return details


def waverec_multilevel_at_once(c: list, w: Wavelet, original_shape: tuple[int, ...]):

    a = c[0]
    d = c[1:]
    d.reverse()
    res = OffsetMatrix(np.zeros((1, 1)), np.array([0, 0]))
    m = w.m
    wmasks = [OffsetMatrix(wmask.matrix * m, wmask.offset) for wmask in w.g]
    cur_M = w.M.copy()
    for i, di in enumerate(d):
        for j, dij in enumerate(di):
            res += subdivision(dij, wmasks[j], cur_M)
            #print(wmasks[j].matrix)
            wmasks[j] = subdivision(wmasks[j], w.h, w.M)
            #print(wmasks[j].matrix)
            wmasks[j].matrix = wmasks[j].matrix * m
            cur_M @= w.M

    mask_h = OffsetMatrix(w.h.matrix * m, w.h.offset)
    cur_M = w.M.copy()
    for i in range(len(d)-1):
        mask_h = subdivision(mask_h, w.h, w.M)
        mask_h.matrix = mask_h.matrix * m
        cur_M @= w.M
    res += subdivision(a, mask_h, cur_M)


    #res = res.matrix[*tuple(map(slice, tuple(-res.offset), tuple(-res.offset+original_shape)))]
    return res








#data = OffsetMatrix(iio.imread('test/lenna.bmp'), np.array([0,0]))
data = OffsetMatrix(28 * np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]), np.array([0,0]))

M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25], [0.5], [0.25]]), np.array([0,1]))
g = (OffsetMatrix(np.array([[-0.125], [-0.25], [0.75], [-0.25], [-0.125]]), np.array([0,3])),)
hdual = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual = (OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,2])),)

hdual_conj = OffsetMatrix(np.array([[-0.125], [0.25], [0.75], [0.25], [-0.125]]),np.array([0,2]))
gdual_conj = (OffsetMatrix(np.array([[-0.25], [0.5], [-0.25]]),np.array([0,0])),)

w = Wavelet(h, g, hdual, gdual, M, np.abs(np.linalg.det(M)))

ci_ = wavedec(data, 1, w)
ci = wavedec_multilevel_at_once(data, w, 1)
#print(ai_.matrix)
res = waverec_multilevel_at_once(ci, w, [5, 5])
print(res.matrix)
ress = waverec(ci_, w, [5, 5])
print(ress)
iio.imwrite('res.png', np.clip(res.matrix, 0, 255).astype(np.uint8))
iio.imwrite('ress.png', np.clip(ci[0].matrix, 0, 255).astype(np.uint8))
iio.imwrite('resss.png', np.clip(ress, 0, 255).astype(np.uint8))
#iio.imwrite('ress_.png', np.clip(ci_[0].matrix, 0, 255).astype(np.uint8))
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
