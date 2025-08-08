import numpy as np
import imageio.v3 as iio

from classic.wave import wavedec, waverec
from metrics import psnr
from offset_matrix import OffsetMatrix
from periodic.wave import wavedec_period, waverec_period
from wavelet import Wavelet

if __name__ == "__main__":
    #data = OffsetMatrix(iio.imread('test/lenna.bmp'), np.array([0,0]))
    data = OffsetMatrix(28. * np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]), np.array([0,0]))

    M = np.array([[1, -1], [1,1]])

    h = OffsetMatrix(np.array([[0.25, 0.5, 0.25]]), np.array([0,-1]))
    g = (OffsetMatrix(np.array([[-0.125, -0.25, 0.75, -0.25, -0.125]]), np.array([0,-1])),)
    hdual = OffsetMatrix(np.array([[-0.125, 0.25, 0.75, 0.25, -0.125]]), np.array([0,-2]))
    gdual = (OffsetMatrix(np.array([[-0.25, 0.5, -0.25]]),np.array([0,0])),)



    w = Wavelet(h, g, hdual, gdual, M, np.abs(np.linalg.det(M)))

    ci_ = wavedec(data, 2, w)
    ci = wavedec_period(data, w, 2)
    #clamp(ci)

    res_classic = waverec(ci_, w, np.array([5, 5]))
    ress = waverec_period(ci, w, np.array([5, 5]))
    print(res_classic)
    print(ress.matrix)
    #print(data.matrix)
    #ress = waverec(ci_, w, [5, 5])
    iio.imwrite('res.png', np.clip(ress.matrix, 0, 255).astype(np.uint8))
    print("PSNR:", psnr(data.matrix, ress.matrix))
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
