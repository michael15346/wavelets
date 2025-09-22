import json

import numpy as np

from db import createWaveletFromContent
from metrics import psnr
from offset_tensor import OffsetTensor
import imageio.v3 as iio

from multilevel.wave import wavedec_multilevel_at_once, waverec_multilevel_at_once
from periodic.wave import wavedec_period, waverec_period
from classic.wave import wavedec, waverec
from wavelet import Wavelet


def roundtrip(input, output):
    data = OffsetTensor(iio.imread(input).mean(axis=2), np.array([0, 0]))
    # data = OffsetTensor(28. * np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11]]), np.array([0,0]))

    with open("WaveDB.json", 'r') as j:
        contents = json.loads(j.read())
    content = contents[22]
    w = createWaveletFromContent(content)

    print(w)
    # ci_ = wavedec(data, 1, w)
    ci = wavedec_period(data, w, 5)
    # entropy = uniform_entropy(ci)

    # clamp(ci)
    # ci = roundtrip_kmeans(ci, 8)

    # res_classic = waverec(ci_, w, np.array([5, 5]))
    ress = waverec_period(ci, w, np.array(data.tensor.shape))
    # print(res_classic)
    print("recovered:", ress.tensor)
    print("original:", data.tensor)
    #print("diff:", ress.tensor - data.tensor)
    # ress = waverec(ci_, w, [5, 5])
    iio.imwrite(output, np.clip(ress.tensor, 0, 255).astype(np.uint8))
    iio.imwrite('diff.png', np.clip(np.abs(ress.tensor - data.tensor) * 3e15, 0, 255).astype(np.uint8))
    print("PSNR:", psnr(data.tensor, ress.tensor))
    # print("entropy:", entropy)
    # iio.imwrite('ress.png', np.clip(ci[0], 0, 255).astype(np.uint8))
    # iio.imwrite('resss.png', np.clip(ress, 0, 255).astype(np.uint8))
    # iio.imwrite('ress_.png', np.clip(ci_[0].matrix, 0, 255).astype(np.uint8))
    # for dd in d:
    #    for ddd in dd:
    #
    #
    # iio.imwrite('a.png', np.clip(ai.matrix, 0, 255).astype(np.uint8))
    # for i, di in enumerate(d):
    #    for j, dij in enumerate(di):
    #        iio.imwrite(f'd{i}-{j}.png', np.clip(dij.matrix * (w.m ** (5 -i )), 0, 255).astype(np.uint8))
    #
    # a = waverec(ai, d, w, data.matrix.shape)

    # iio.imwrite('restored.png', np.clip(a, 0, 255).astype(np.uint8))
