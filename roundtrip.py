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
import cv2

from multilevel.wave import wavedec_multilevel_at_once, waverec_multilevel_at_once
from periodic.wave import wavedec_period, waverec_period
from classic.wave import wavedec, waverec
from quant import encode_uniform, decode_uniform
from wavelet import Wavelet


def roundtrip(ii, oo):
    data = iio.imread(ii).astype(np.float32)
    #data = OffsetTensor(, np.array([0, 0]))
    # data = OffsetTensor(28. * np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11]]), np.array([0,0]))
    dataYCC = cv2.cvtColor(data, cv2.COLOR_RGB2YCR_CB)
    data_restored = np.zeros_like(data)

    with open("WaveDB.json", 'r') as j:
        contents = json.loads(j.read())

    print(contents[26])
    for content_idx, content in enumerate(contents):

        w = createWaveletFromContent(content)
        for ch in (0, 1, 2):
            level = 5
            data_ch = OffsetTensor(np.rint(dataYCC[:, :, ch]).astype(int), np.array([0, 0]))
            ci = wavedec_period(data_ch, w, level)

            n_cluster = 8
            quantized, quantized_downs, n_min, n_max, downs_min, downs_max = encode_uniform(ci, n_cluster)
            ci_restored = decode_uniform(quantized, quantized_downs, n_min, n_max, downs_min, downs_max, n_cluster, w, data_ch.tensor.shape, level)

            # try:
            #     os.remove('test.ezw')
            # except OSError:
            #     pass

            #encodeEZW(data,  w, 5,'test.ezw')
            #ress = decodeEZW('test.ezw', w)

            ress = waverec_period(ci_restored, w, np.array(data_ch.tensor.shape))
            data_restored[:, :, ch] = ress.tensor
        data_restoredRGB = cv2.cvtColor(data_restored, cv2.COLOR_YCR_CB2RGB)
        iio.imwrite(oo, np.clip(data_restoredRGB, 0, 255).astype(np.uint8))
        iio.imwrite('diff.png', np.clip(np.abs(data_restoredRGB - data), 0, 255).astype(np.uint8))
        psnr_res = psnr(data.astype(np.float64), data_restoredRGB.astype(np.float64))
        if psnr_res < 300:
            print(f"Low PSNR! Content:{content_idx}, PSNR={psnr_res}")
        print('success')
        input()
