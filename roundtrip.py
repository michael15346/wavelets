import json
from copy import deepcopy

import numpy as np

from db import createWaveletFromContent
from ezw.wave import wavedec_ezw, waverec_ezw
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
    for content_idx, content in enumerate(contents):
        w = createWaveletFromContent(content)
        ci = wavedec_ezw(data, w, 5)
        ress = waverec_ezw(ci, w, np.array(data.tensor.shape))
        iio.imwrite(output, np.clip(ress.tensor, 0, 255).astype(np.uint8))
        iio.imwrite('diff.png', np.clip(np.abs(ress.tensor - data.tensor) * 3e15, 0, 255).astype(np.uint8))
        psnr_res = psnr(data.tensor, ress.tensor)
        if psnr_res < 300:
            print(f"Low PSNR! Content:{content_idx}, PSNR={psnr_res}")
        #print("PSNR:", psnr(data.tensor, ress.tensor))
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
