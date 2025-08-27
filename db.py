from itertools import product as direct_product

import numpy as np

from classic.wave import convolve
from offset_tensor import OffsetTensor
from utils import CoefCoords2OffsetMatrix, OffsetMatrix2CoefCoords
from wavelet import Wavelet


def createWaveletFromContent(content):
    M = np.array(content['DilationMatrix'])
    coeffs = np.array(content['Mask']['Coeffs']) / content['Mask']['CoeffsDenominator']
    coords = np.array(content['Mask']['Coords'])
    h = CoefCoords2OffsetMatrix((coeffs, coords))

    coeffs = np.array(content['DualMask']['Coeffs']) / content['DualMask']['CoeffsDenominator']
    coords = np.array(content['DualMask']['Coords'])
    hdual = CoefCoords2OffsetMatrix((coeffs, coords))

    g = []
    for wmask in content['WaveletMasks']:
        coeffs = np.array(content['WaveletMasks'][wmask]['Coeffs']) / content['WaveletMasks'][wmask][
            'CoeffsDenominator']
        coords = np.array(content['WaveletMasks'][wmask]['Coords'])
        g.append(CoefCoords2OffsetMatrix((coeffs, coords)))
    g = tuple(g)

    gdual = []
    for wmask in content['DualWaveletMasks']:
        coeffs = np.array(content['DualWaveletMasks'][wmask]['Coeffs']) / content['DualWaveletMasks'][wmask][
            'CoeffsDenominator']
        coords = np.array(content['DualWaveletMasks'][wmask]['Coords'])
        gdual.append(CoefCoords2OffsetMatrix((coeffs, coords)))
    gdual = tuple(gdual)

    return Wavelet(h, g, hdual, gdual, M, abs(np.linalg.det(M)))


def DeltaN(n, d):
    cube = direct_product(range(n), repeat=d)
    result = []
    for coord in cube:
        if sum(coord) < n:
            result.append(list(coord))
    return result


def PlaneN(n, d):
    cube = direct_product(range(n), repeat=d)
    result = []
    for coord in cube:
        if sum(coord) == n - 1:
            result.append(list(coord))
    return result


def MatrixOfPowers(aCoords, bPowers):
    matrix = np.zeros((len(aCoords), len(bPowers)))
    for i, ai in enumerate(aCoords):
        for j, bj in enumerate(bPowers):
            matrix[i, j] = np.prod(np.power(ai, bj))
    return matrix


def VM(wmask: OffsetTensor, tolerance=1e-6, max_vm=None):
    """
    Calculate the order of vanishing moments a wavelet filter.

    Args:
        wmask (OffsetTensor): Wavelet filter in the OffsetTensor form.
        tolerance (float, optional): Numerical tolerance to consider sum as zero. Defaults to 1e-6.
        max_vm (int, optional): Maximum order to check. Defaults to the length of coefficients.

    Returns:
        int: Order of vanishing moments.
    """
    (coefs, coords) = OffsetMatrix2CoefCoords(wmask)
    d = len(wmask.offset)
    vm = 1
    if max_vm is None:
        max_vm = len(coefs)

    while vm <= max_vm:
        bPowers = PlaneN(vm, d)
        MatrixOfPowers(coords, bPowers)
        vm_check = np.dot(coefs, MatrixOfPowers(coords, bPowers))

        if sum(abs(vm_check)) < tolerance:
            vm += 1
        else:
            break

    return vm - 1


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

    (coefs, coords) = OffsetMatrix2CoefCoords(mask)
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
            polyphases.append(CoefCoords2OffsetMatrix(([0], [np.zeros(dim, dtype=np.int32)])))
        else:
            polyphases.append(CoefCoords2OffsetMatrix((poly_coefs, poly_coords)))

    return polyphases


def SR(mask: OffsetTensor, M, digits=None, tolerance=1e-6, max_sr=None):
    """
    Calculate the order of sum rule of filter.

    Args:
        mask (OffsetTensor): Wavelet filter in the OffsetTensor form.
        M: matrix dilation.
        digits: the set of digits
        tolerance (float, optional): Numerical tolerance to consider sum as zero. Defaults to 1e-6.
        max_sr (int, optional): Maximum order to check. Defaults to the length of coefficients.

    Returns:
        int: Order of sum rule.
    """
    if digits is None:
        digits = SetOfDigitsFinder(M)
    polyphases = Mask2Polyphase(mask, M)
    d = len(M)
    sr = 1
    if max_sr is None:
        max_sr = 20

    while sr <= max_sr:
        bPowers = PlaneN(sr, d)
        sr_check = None
        sr_stop = False
        for digit, polyphase in zip(digits, polyphases):
            (poly_coefs, poly_coords) = OffsetMatrix2CoefCoords(polyphase)
            poly_coords = (poly_coords @ M.T) + digit
            if sr_check is None:
                sr_check = np.dot(poly_coefs, MatrixOfPowers(poly_coords, bPowers))
            else:
                if not np.allclose(sr_check, np.dot(poly_coefs, MatrixOfPowers(poly_coords, bPowers)), atol=tolerance):
                    sr_stop = True
        if sr_stop:
            break
        sr = sr + 1
        if sr > max_sr:
            break

    return sr - 1


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
            tmp = OffsetTensor(np.array([[0]]), np.array(np.zeros(dim, dtype=np.int32)))
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


def checkContent(content):
    w = createWaveletFromContent(content)
    if SR(w.h, w.M) == content['Mask']['SR']:
        print("Mask SR OK")
    else:
        print("Mask SR wrong: by checker SR = ", SR(w.h, w.M),
              ", by Database SR = ", content['Mask']['SR'])
    if SR(w.hdual, w.M) == content['DualMask']['SR']:
        print("Dual Mask SR OK")
    else:
        print("Dual Mask SR wrong: by checker SR = ", SR(w.h, w.M),
              ", by Database SR = ", content['DualMask']['SR'])

    for wmask in w.g:
        if VM(wmask) >= content['WaveletMasksVM']:
            print("Wavelet Mask VM OK")
        else:
            print("Wavelet Mask VM wrong: by checker VM = ", VM(wmask),
                  ", by Database VM = ", content['WaveletMasksVM'])

    for wmask in w.gdual:
        if VM(wmask) >= content['DualWaveletMasksVM']:
            print("Dual Wavelet Mask VM OK")
        else:
            print("Dual Wavelet Mask VM wrong: by checker VM = ", VM(wmask),
                  ", by Database VM = ", content['DualWaveletMasksVM'])
    PRP_valid, _ = PRP_check(w)
    if PRP_valid:
        print("PRP check OK")
    else:
        print("PRP check Wrong")