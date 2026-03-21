import numpy as np

from offset_tensor import OffsetTensor

def to_python_vect(coords, offset):
    return (coords - offset).T

def OffsetTensor2CoefCoords(a: OffsetTensor):
    dim = len(a.offset)
    non_zero_coords = np.nonzero(a.tensor)
    coef = a.tensor[non_zero_coords]
    coords = np.array(non_zero_coords).T + a.offset
    if len(coef) == 0:
        coef, coords = [0], [np.zeros(dim, dtype = np.int32)]
    return np.array(coef), np.array(coords)

def CoefCoords2OffsetTensor(a):  # a filter in coef-coords form
    coef, coords = a
    min_corner = np.min(np.transpose(coords), axis=1)
    max_corner = np.max(np.transpose(coords), axis=1)
    shape = max_corner - min_corner + np.ones(min_corner.shape, dtype = np.int32)
    tensor = np.zeros(shape)
    for ind, coord in enumerate(coords):
        tensor[*tuple(coord - min_corner)] = coef[ind]
    return OffsetTensor(tensor, min_corner)

def ci_size(a):
    d = a[1:]
    a = a[0]
    total = a.size
    for di in d:
        for dj in di:
            total += dj.size
    return total