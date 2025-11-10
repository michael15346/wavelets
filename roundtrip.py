import json
import os
from copy import deepcopy

import numpy as np

from db import createWaveletFromContent
from ezw.wave import wavedec_ezw, waverec_ezw
from ezw.codec import encodeEZW, decodeEZW
from metrics import psnr
from offset_tensor import OffsetTensor
import imageio.v3 as iio

from multilevel.wave import wavedec_multilevel_at_once, waverec_multilevel_at_once
from periodic.wave import wavedec_period, waverec_period
from classic.wave import wavedec, waverec
from wavelet import Wavelet


def roundtrip(input, output):
    data = OffsetTensor(iio.imread(input).mean(axis=2).astype(np.uint8), np.array([0, 0]))
    # data = OffsetTensor(28. * np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11]]), np.array([0,0]))

    with open("WaveDB.json", 'r') as j:
        contents = json.loads(j.read())
    for content_idx, content in enumerate(contents):
        w = createWaveletFromContent(content)
        #ci = wavedec_period(data, w, 5)
        try:
            os.remove('test.ezw')
        except OSError:
            pass

        encodeEZW(data,  w, 5,'test.ezw')
        ress = decodeEZW('test.ezw', w)
        # ress = waverec_period(ci, w, np.array(data.tensor.shape))
        iio.imwrite(output, np.clip(ress.tensor, 0, 255).astype(np.uint8))
        iio.imwrite('diff.png', np.clip(np.abs(ress.tensor - data.tensor) * 3e15, 0, 255).astype(np.uint8))
        psnr_res = psnr(data.tensor.astype(np.uint8).astype(np.float64), ress.tensor.astype(np.float64))
        if psnr_res < 300:
            print(f"Low PSNR! Content:{content_idx}, PSNR={psnr_res}")
