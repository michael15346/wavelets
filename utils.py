from copy import deepcopy
import numpy as np
from offset_matrix import OffsetMatrix


def to_python(x, y, offset):
    return x - offset[0], y - offset[1]


def to_coord(row, col, offset):
    return offset[0]+col, offset[1]-row


def to_python_vect(coords, offset):
    return coords.T[0] - offset[0], coords.T[1]-offset[1]


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

