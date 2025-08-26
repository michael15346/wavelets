import numpy as np
import imageio.v3 as iio

from classic.wave import wavedec, waverec
from metrics import psnr
from multilevel.wave import waverec_multilevel_at_once, wavedec_multilevel_at_once
from offset_tensor import OffsetTensor
from periodic.wave import wavedec_period, waverec_period
from quant import roundtrip_kmeans, uniform_roundtrip, uniform_entropy
from wavelet import Wavelet

if __name__ == "__main__":
    data = OffsetTensor(iio.imread('test/lenna.bmp'), np.array([0, 0]))
    #data = OffsetTensor(28. * np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11]]), np.array([0,0]))

    M = np.array([[1, -1], [1,1]])

    h = OffsetTensor(np.array([[0.25, 0.5, 0.25]]), np.array([0, -1]))
    g = (OffsetTensor(np.array([[-0.125, -0.25, 0.75, -0.25, -0.125]]), np.array([0, -1])),)
    hdual = OffsetTensor(np.array([[-0.125, 0.25, 0.75, 0.25, -0.125]]), np.array([0, -2]))
    gdual = (OffsetTensor(np.array([[-0.25, 0.5, -0.25]]), np.array([0, 0])),)



    w = Wavelet(h, g, hdual, gdual, M, np.abs(np.linalg.det(M)))

    #ci_ = wavedec(data, 1, w)
    ci = wavedec_period(data, w, 1)
    entropy = uniform_entropy(ci)

    #clamp(ci)
    ci = uniform_roundtrip(ci)

    #res_classic = waverec(ci_, w, np.array([5, 5]))
    ress = waverec_period(ci, w, np.array(data.tensor.shape))
    #print(res_classic)
    print("recovered:", ress.tensor)
    print("original:", data.tensor)
    #ress = waverec(ci_, w, [5, 5])
    iio.imwrite('res.png', np.clip(ress.tensor, 0, 255).astype(np.uint8))
    print("PSNR:", psnr(data.tensor, ress.tensor))
    print("entropy:", entropy)
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
