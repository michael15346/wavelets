from itertools import product as direct_product

import numpy as np

from classic.wave import convolve
from offset_tensor import OffsetTensor
from utils import CoefCoords2OffsetTensor, OffsetTensor2CoefCoords
from wavelet import Wavelet


def createWaveletFromContent(content):
    M = np.array(content['DilationMatrix'], dtype=int)
    coeffs = np.array(content['Mask']['Coeffs']) / content['Mask']['CoeffsDenominator']
    coords = np.array(content['Mask']['Coords'])
    h = CoefCoords2OffsetTensor((coeffs, coords))

    coeffs = np.array(content['DualMask']['Coeffs']) / content['DualMask']['CoeffsDenominator']
    coords = np.array(content['DualMask']['Coords'])
    hdual = CoefCoords2OffsetTensor((coeffs, coords))

    g = []
    for wmask in content['WaveletMasks']:
        coeffs = np.array(content['WaveletMasks'][wmask]['Coeffs']) / content['WaveletMasks'][wmask][
            'CoeffsDenominator']
        coords = np.array(content['WaveletMasks'][wmask]['Coords'])
        g.append(CoefCoords2OffsetTensor((coeffs, coords)))
    g = tuple(g)

    gdual = []
    for wmask in content['DualWaveletMasks']:
        coeffs = np.array(content['DualWaveletMasks'][wmask]['Coeffs']) / content['DualWaveletMasks'][wmask][
            'CoeffsDenominator']
        coords = np.array(content['DualWaveletMasks'][wmask]['Coords'])
        gdual.append(CoefCoords2OffsetTensor((coeffs, coords)))
    gdual = tuple(gdual)

    return Wavelet(h, g, hdual, gdual, M, np.rint(abs(np.linalg.det(M))).astype(int))

def SetOfDigitsFinder(M):
    dim = len(M)
    Minv = np.linalg.inv(M)

    min_corner = np.zeros(dim, dtype=np.int32)
    max_corner = np.ones(dim, dtype=np.int32)
    cube_corners = M @ np.array(list(direct_product(*zip(list(min_corner), list(max_corner))))).T

    all_min_corner = np.ceil(np.min(cube_corners, axis=1)).astype(np.int32)
    all_max_corner = np.floor(np.max(cube_corners, axis=1)).astype(np.int32)
    cube_inside_coords = np.mgrid[
        *tuple(map(slice, tuple(all_min_corner), tuple(all_max_corner + np.ones(dim, dtype=np.int32))))].reshape(dim,
                                                                                                                 -1)

    Mcoords = (Minv @ cube_inside_coords).T
    mask = np.all(np.floor(Mcoords).astype(np.int32) == np.zeros(dim, dtype=np.int32), axis=1)
    return cube_inside_coords.T[mask]


def Mask2Polyphase(mask: OffsetTensor, M, digits=None):
    if digits is None:
        digits = SetOfDigitsFinder(M)

    (coefs, coords) = OffsetTensor2CoefCoords(mask)
    m = round(np.abs(np.linalg.det(M)))
    Minv_pre = np.rint((m * np.linalg.inv(M))).astype(np.int32)
    dim = len(M)

    polyphases = []
    for digit in digits:
        pre_poly_coords = Minv_pre @ (coords - digit).T
        mask = np.all(np.mod(pre_poly_coords, m) == 0, axis=0)
        poly_coords = pre_poly_coords.T[mask] // m
        poly_coefs = coefs[mask]
        if len(poly_coefs) == 0:
            polyphases.append(CoefCoords2OffsetTensor(([0], [np.zeros(dim, dtype=np.int32)])))
        else:
            polyphases.append(CoefCoords2OffsetTensor((poly_coefs, poly_coords)))

    return polyphases


def PRP_check(w: Wavelet, digits=None, tolerance=1e-6):
    if digits is None:
        digits = SetOfDigitsFinder(w.M)
    polyphase_matrix = []
    dual_polyphase_matrix = []
    dim = len(w.M)
    polyphase_matrix.append(Mask2Polyphase(w.h, w.M, digits))
    dual_polyphase_matrix.append(Mask2Polyphase(w.hdual, w.M, digits))
    for wmask in w.g:
        polyphase_matrix.append(Mask2Polyphase(wmask, w.M, digits))
    for dwmask in w.gdual:
        dual_polyphase_matrix.append(Mask2Polyphase(dwmask, w.M, digits))

    PRP_matrix = []
    PRP_valid = True
    m = int(w.m)
    for i in range(m):
        PRP_matrix.append([])
        for j in range(m):
            tmp = OffsetTensor(np.zeros([1] * dim), np.array(np.zeros(dim, dtype=np.int32)))
            for k in range(len(w.g) + 1):
                tmp = tmp + convolve(polyphase_matrix[k][i], dual_polyphase_matrix[k][j].conjugate())

            if i == j:
                tmp_check = np.zeros_like(tmp.tensor)
                tmp_check[*(-np.array(tmp.offset))] = 1 / w.m
                if abs(np.sum(np.abs(tmp.tensor - tmp_check))) > tolerance:
                    PRP_valid = False
                    print("False:", i, ' ', j, ' ', tmp.tensor)
            else:
                if abs(np.sum(np.abs(tmp.tensor))) > tolerance:
                    PRP_valid = False
                    print("False:", i, ' ', j, ' ', tmp.tensor)

            PRP_matrix[i].append(tmp)
    return PRP_valid, PRP_matrix
