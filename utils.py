import numpy as np
from PIL import Image

from offset_tensor import OffsetTensor

def to_python(x, y, offset):
    return x - offset[0], y - offset[1]

def to_python_vect(coords, offset):
    return (coords - offset).T

# def OffsetTensor2CoefCoords(a: OffsetTensor):
#     shape = a.tensor.shape
#     offset = a.offset
#     coef = []
#     coords = []
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             if a.tensor[i, j] != 0:
#                 coef.append(a.tensor[i, j])
#                 coords.append([offset[0]+i, offset[1]+j])
#     if len(coef) == 0:
#         coef, coords = [0], [[0,0]]
#     return np.array(coef), np.array(coords)
# 
# 
# def CoefCoords2OffsetTensor(a):  # a filter in coef-coords form
#     coef = a[0]
#     coords = a[1]
#     x_left, y_bottom = list(np.min(np.transpose(coords), axis=1))
#     x_right, y_up = list(np.max(np.transpose(coords), axis=1))
#     shape = [x_right-x_left+1, y_up-y_bottom+1]
#     offset = np.array([x_left, y_bottom])
#     matrix = np.zeros(shape)
#     for ind, coord in enumerate(coords):
#         row, col = to_python(coord[0], coord[1], offset)
#         matrix[row, col] = coef[ind]
#     return OffsetTensor(matrix, offset)

def OffsetTensor2CoefCoords(a: OffsetTensor):
    dim = len(a.offset)
    non_zero_coords = np.nonzero(a.tensor)
    coef = a.tensor[non_zero_coords]
    coords = np.array(non_zero_coords).T + a.offset
    if len(coef) == 0:
        coef, coords = [0], [np.zeros(dim, dtype = np.int32)]
    return (np.array(coef), np.array(coords))

def CoefCoords2OffsetTensor(a):  # a filter in coef-coords form
    coef, coords = a
    min_corner = np.min(np.transpose(coords), axis=1)
    max_corner = np.max(np.transpose(coords), axis=1)
    shape = max_corner - min_corner + np.ones(min_corner.shape, dtype = np.int32)
    tensor = np.zeros(shape)
    for ind, coord in enumerate(coords):
        tensor[*tuple(coord - min_corner)] = coef[ind]
    return OffsetTensor(tensor, min_corner)

def clamp(a):
    d = a[1:]
    a = a[0]
    clamped = np.sum(np.abs(a.tensor) <= 10)
    a.tensor = np.where(np.abs(a.tensor) > 10, a.tensor, 0)
    total = a.tensor.size
    for di in d:
        for dj in di:
            total += dj.tensor.size
            clamped += np.sum(np.abs(dj.tensor) <= 10)
            dj.tensor = np.where(np.abs(dj.tensor) > 10, dj.tensor, 0)

CONV_MAT = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.4188688, -0.081312]]).T
INV_CONV_MAT = np.linalg.inv(CONV_MAT)


def resize(img, M, N):
    return np.array(Image.fromarray(img).resize((N, M), resample=Image.BILINEAR))

def RGB2YCbCr(im_rgb):
    im_ycbcr = np.array([-128, 0, 0]) + im_rgb.tensor @ CONV_MAT
    im_ycbcr = np.where(im_ycbcr > 127, 127, im_ycbcr)
    im_ycbcr = np.where(im_ycbcr < -128, -128, im_ycbcr)

    return OffsetTensor(im_ycbcr, np.zeros_like(im_ycbcr.shape))


def YCbCr2RGB(im_ycbcr):
    im_rgb = (np.array([128, 0, 0]) + im_ycbcr) @ INV_CONV_MAT
    im_rgb = np.where(im_rgb > 255, 255, im_rgb)
    im_rgb = np.where(im_rgb < 0, 0, im_rgb)
    return OffsetTensor(im_rgb, np.zeros_like(im_rgb.shape))