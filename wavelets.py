import numpy as np
import scipy.signal
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim
import imageio.v3 as iio
from dataclasses import dataclass
from math import ceil, floor


class OffsetMatrix:
    matrix: np.ndarray
    offset: np.ndarray

    def __init__(self, matrix: np.ndarray, offset):
        self.matrix = matrix
        if (matrix.ndim == 3):
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
    new_offset = np.array([a.offset[0] + b.offset[0], a.offset[1] + b.offset[1]])
    new_matrix = scipy.signal.convolve2d(a.matrix, b.matrix, mode = 'full', boundary = 'fill')
    return OffsetMatrix(new_matrix, new_offset)

def convolve_dummy(shape, offset, mask_shape, mask_offset):
    new_offset = np.array([offset[0] + mask_offset[0], offset[1] + mask_offset[1]])
    new_shape = np.array([shape[0] + (mask_shape[0] - 1), shape[1] + (mask_shape[1] - 1)])
    return (new_shape, new_offset)

def transition(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample(convolve(a, mask), M)


def transition_dummy(shape, offset, mask_shape, mask_offset, M):
    (mask_shape, mask_offset) = OffsetMatrixConjugate_dummy(mask_shape, mask_offset)
    (shape, offset) = convolve_dummy(shape, offset, mask_shape, mask_offset)
    return (shape, offset)

def transition_vector(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    mask = OffsetMatrixConjugate(mask)
    return downsample_vector(convolve(a, mask), M)


def subdivision(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray):
    u = upsample(a,M)
    c = convolve(u, mask)
    return c#convolve(upsample(a, M), mask)

def subdivision_vector(a: OffsetMatrix, mask: OffsetMatrix, M: np.ndarray, original_shape, original_offset):
    u = upsample_vector(a,M, original_shape, original_offset)
    c = convolve(u, mask)
    return c#convolve(upsample(a, M), mask)

def subdivision_dummy(matrix_shape, matrix_offset, mask_shape, mask_offset, M: np.ndarray):
    (shape, offset) = upsample_dummy(matrix_shape, matrix_offset ,M)
    (offset_, dummy_) = convolve_dummy(shape, offset, mask_shape, mask_offset)
    return (offset_, dummy_)

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
    x1 = -ai.offset[0]
    y1 = -ai.offset[1]
    x2 = x1 + original_shape[0]
    y2 = y1 + original_shape[1]
    ai = ai.matrix[x1:x2, y1:y2]
    return ai


def clamp(a):
    d = a[1:]
    a = a[0]
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
    a_conj.offset = np.array([-(a.offset[0] + a.matrix.shape[0] - 1), -(a.offset[1] + a.matrix.shape[1] - 1)])


    return a_conj

def OffsetMatrixConjugate_dummy(a_shape, a_offset):
    offset = np.array([-(a_offset[0] + a_shape[0] - 1), -(a_offset[1] + a_shape[1] - 1)])
    return (a_shape, offset)

def to_python_vect(coords, offset):
    # all_x, all_y
    #return offset[1]-coords.T[1], coords.T[0]-offset[0]
    return coords.T[0] - offset[0], coords.T[1]-offset[1]


def downsample(a: OffsetMatrix, M: np.ndarray):
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


    
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0] + a.matrix.shape[0],
                              a.offset[1]:a.offset[1] + a.matrix.shape[1]].reshape(2, -1)  
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)
    
    downs_coords = list(to_python_vect(downs_coords.T[mask]//m, downsampled.offset))
    downsampled.matrix[downs_coords[0], downs_coords[1]] = a.matrix[lattice_coords[0], lattice_coords[1]]
    return downsampled

def downsample_vector(a: OffsetMatrix, M: np.ndarray):
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


    
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0] + a.matrix.shape[0],
                              a.offset[1]:a.offset[1] + a.matrix.shape[1]].reshape(2, -1)  
    downs_coords = (Minv_pre @ lattice_coords)
    mask = np.all(np.mod(downs_coords, m) == 0, axis=0)
    lattice_coords = to_python_vect(lattice_coords.T[mask], a.offset)
    return a.matrix[lattice_coords[0], lattice_coords[1]]

def downsample_dummy(shape, offset, M: np.ndarray):
    Minv = np.linalg.inv(M)
    x1 = Minv @ np.array([offset[0], offset[1]])
    x2 = Minv @ np.array([offset[0] + shape[0] - 1, offset[1]])
    x3 = Minv @ np.array([offset[0], offset[1] + shape[1] - 1])
    x4 = Minv @ np.array([offset[0] + shape[0] - 1, offset[1] + shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))
    downsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    return (downsampled.matrix.shape, downsampled.offset)

def upsample_vector(a, M: np.ndarray, original_shape, original_offset):
    Minv = np.linalg.inv(M)
    #original_shape = (7, 7)
    # This not only needs to create lattice_coords like original_shape,
    # but also add borders introduced by convolution
    x1 = Minv @ np.array([original_offset[0], original_offset[1]])
    x2 = Minv @ np.array([original_offset[0] + original_shape[0] - 1, original_offset[1]])
    x3 = Minv @ np.array([original_offset[0], original_offset[1] + original_shape[1] - 1])
    x4 = Minv @ np.array([original_offset[0] + original_shape[0] - 1, original_offset[1] + original_shape[1] - 1])
    xmin = ceil(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = floor(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = ceil(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = floor(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((original_shape[0],original_shape[1]), dtype=np.float64), np.array([original_offset[0], original_offset[1]]))

    lattice_coords = np.mgrid[original_offset[0]:original_offset[0] + original_shape[0],
                          original_offset[1]:original_offset[1] + original_shape[1]].reshape(2, -1)
    Minv_pre = np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])
    ups_coords = Minv_pre @ lattice_coords
    m = round(np.abs(np.linalg.det(M)))
    mask = np.all(np.mod(ups_coords, m) == 0, axis=0)
    ups_coords = to_python_vect(lattice_coords.T[mask], original_offset)
    upsampled.matrix[ups_coords[0], ups_coords[1]] = a
    return upsampled


def upsample_dummy(a_shape, a_offset, M: np.ndarray):
    x1 = M @ np.array([a_offset[0], a_offset[1]])
    x2 = M @ np.array([a_offset[0] + a_shape[0] - 1, a_offset[1]])
    x3 = M @ np.array([a_offset[0], a_offset[1] + a_shape[1]-1])
    x4 = M @ np.array([a_offset[0] + a_shape[0]  - 1, a_offset[1] + a_shape[1] - 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    return (upsampled.matrix.shape, upsampled.offset)


def upsample(a: OffsetMatrix, M: np.ndarray):
    x1 = M @ np.array([a.offset[0], a.offset[1]])
    x2 = M @ np.array([a.offset[0] + a.matrix.shape[0] - 1, a.offset[1]])
    x3 = M @ np.array([a.offset[0], a.offset[1] + a.matrix.shape[1]-1])
    x4 = M @ np.array([a.offset[0] + a.matrix.shape[0]  - 1, a.offset[1] + a.matrix.shape[1] - 1])
    xmin = int(min(x1[0], x2[0], x3[0], x4[0]))
    xmax = int(max(x1[0], x2[0], x3[0], x4[0]))
    ymin = int(min(x1[1], x2[1], x3[1], x4[1]))
    ymax = int(max(x1[1], x2[1], x3[1], x4[1]))
    upsampled = OffsetMatrix(np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=np.float64), np.array([xmin, ymin]))
    lattice_coords = np.mgrid[a.offset[0]:a.offset[0]+a.matrix.shape[0],
                              a.offset[1]:a.offset[1]+a.matrix.shape[1]]\
                       .reshape(2, -1)

    ups_coords = M @ lattice_coords
    lattice_coords = to_python_vect(lattice_coords.T, a.offset)
    ups_coords = to_python_vect(ups_coords.T, upsampled.offset)
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
            tmp_list.append(transition_vector(data, m, cur_M.copy()))
        details.append(tmp_list)
        #details.append(list(map(
        #    transition, [data] * len(mask), mask, [cur_M] * len(mask))))
    details.append(transition_vector(data, ref_mask, cur_M))
    details.reverse()


    return details

def wavedec_multilevel_at_once_dummy(data_shape, data_offset, w: Wavelet, level: int):
    mask = [[ww.matrix.shape, ww.offset] for ww in w.gdual]
    masks = [mask]

    for i in range(1, level):
        gmasks = []
        for gdual in w.gdual:
            cur_mask = w.hdual
            cur_mask_shape = cur_mask.matrix.shape
            cur_mask_offset = cur_mask.offset
            cur_M = w.M.copy()
            for j in range(i-1, 0, -1):
                cur_mask_shape, cur_mask_offset = subdivision_dummy(w.hdual.matrix.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
                cur_M @= w.M
            wave_mask_shape, wave_mask_offset = subdivision_dummy(gdual.matrix.shape, gdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
            gmasks.append([wave_mask_shape, wave_mask_offset])
        masks.append(gmasks)
    # !!!
    if level > 1:
        ref_mask_shape, ref_mask_offset = subdivision_dummy(w.hdual.matrix.shape, w.hdual.offset, cur_mask_shape, cur_mask_offset, cur_M)
    else:
        ref_mask_shape, ref_mask_offset = w.hdual.matrix.shape, w.hdual.offset
    details = []
    cur_M = np.eye(w.M.shape[0], dtype=int)
    for mask in masks:
        cur_M @= w.M
        tmp_list = list()
        for m in mask:
            shape_, offset_ = transition_dummy(data_shape, data_offset, m[0], m[1], cur_M.copy()) 
            tmp_list.append([shape_, offset_])
        details.append(tmp_list)
        #details.append(list(map(
        #    transition, [data] * len(mask), mask, [cur_M] * len(mask))))
    shape_, offset_ = transition_dummy(data_shape, data_offset, ref_mask_shape, ref_mask_offset, cur_M)
    details.append([[shape_, offset_]])
    details.reverse()

    return details

def waverec_multilevel_at_once(c: list, w: Wavelet, original_shape, original_offset=np.array([0,0])):


    a = c[0]
    d = c[1:]
    og_s_o = wavedec_multilevel_at_once_dummy(original_shape, original_offset, w, len(d))
    d.reverse()
    res = OffsetMatrix(np.zeros((1, 1)), np.array([0, 0]))
    m = w.m
    wmasks = [OffsetMatrix(wmask.matrix * m, wmask.offset) for wmask in w.g]
    cur_M = w.M.copy()
    for i, di in enumerate(d):
        for j, dij in enumerate(di):
            res += subdivision_vector(dij, wmasks[j], cur_M, og_s_o[len(d) - i][0][0], og_s_o[len(d) - i][0][1])
            wmasks[j] = subdivision(wmasks[j], w.h, w.M)
            wmasks[j].matrix = wmasks[j].matrix * m
            cur_M @= w.M

    mask_h = OffsetMatrix(w.h.matrix * m, w.h.offset)
    cur_M = w.M.copy()
    for i in range(len(d)-1):
        mask_h = subdivision(mask_h, w.h, w.M)
        mask_h.matrix = mask_h.matrix * m
        cur_M @= w.M

    res += subdivision_vector(a, mask_h, cur_M, og_s_o[0][0][0], og_s_o[0][0][1])


    #res = res.matrix[*tuple(map(slice, tuple(-res.offset), tuple(-res.offset+original_shape)))]
    res.matrix = res.matrix[-res.offset[0]:-res.offset[0] + original_shape[0], -res.offset[1] :-res.offset[1] + original_shape[1]]
    return res








data = OffsetMatrix(iio.imread('test/lenna.bmp'), np.array([0,0]))
#data = OffsetMatrix(28 * np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]), np.array([0,0]))

M = np.array([[1, -1], [1,1]])

h = OffsetMatrix(np.array([[0.25, 0.5, 0.25]]), np.array([0,-1]))
g = (OffsetMatrix(np.array([[-0.125, -0.25, 0.75, -0.25, -0.125]]), np.array([0,-1])),)
hdual = OffsetMatrix(np.array([[-0.125, 0.25, 0.75, 0.25, -0.125]]), np.array([0,-2]))
gdual = (OffsetMatrix(np.array([[-0.25, 0.5, -0.25]]),np.array([0,0])),)



w = Wavelet(h, g, hdual, gdual, M, np.abs(np.linalg.det(M)))

#ci_ = wavedec(data, 3, w)
ci = wavedec_multilevel_at_once(data, w, 10)
#clamp(ci)
ress = waverec_multilevel_at_once(ci, w, np.array([512, 512]))

#ress = waverec(ci_, w, [5, 5])
iio.imwrite('res.png', np.clip(ress.matrix, 0, 255).astype(np.uint8))
#iio.imwrite('ress.png', np.clip(ci[0], 0, 255).astype(np.uint8))
#iio.imwrite('resss.png', np.clip(ress, 0, 255).astype(np.uint8))
#iio.imwrite('ress_.png', np.clip(ci_[0].matrix, 0, 255).astype(np.uint8))
#for dd in d:
#    for ddd in dd:
#
#
#iio.imwrite('a.png', np.clip(ai.matrix, 0, 255).astype(np.uint8))
#for i, di in enumerate(d):
#    for j, dij in enumerate(di):
#        iio.imwrite(f'd{i}-{j}.png', np.clip(dij.matrix * (w.m ** (5 -i )), 0, 255).astype(np.uint8))
#
#a = waverec(ai, d, w, data.matrix.shape)

#iio.imwrite('restored.png', np.clip(a, 0, 255).astype(np.uint8))
