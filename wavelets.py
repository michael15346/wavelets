import numpy as np
import scipy.signal
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim
import imageio.v3 as iio
from dataclasses import dataclass
from math import ceil, floor

from multilevel.wave import wavedec_multilevel_at_once, waverec_multilevel_at_once
from offset_matrix import OffsetMatrix
from wavelet import Wavelet

if __name__ == "__main__":
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
