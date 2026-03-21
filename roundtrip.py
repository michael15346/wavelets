import json

import numpy as np

from batched import wavedec_period_batched, waverec_period_batched
from db import createWaveletFromContent
from metrics import psnr
from offset_tensor import OffsetTensor
import imageio.v3 as iio
import cv2



def roundtrip(ii, oo):
    data = iio.imread(ii).astype(np.float32)
    dataYCC = cv2.cvtColor(data, cv2.COLOR_RGB2YCR_CB)
    data_restored = np.zeros_like(data)

    with open("WaveDB.json", 'r') as j:
        contents = json.loads(j.read())

    for content_idx, content in enumerate(contents):

        w = createWaveletFromContent(content)
        for ch in (0, 1, 2):
            level = 5
            data_ch = OffsetTensor(np.rint(dataYCC[:, :, ch]).astype(int), np.array([0, 0]))
            ci = wavedec_period_batched(data_ch, w, level)
            ress = waverec_period_batched(ci, w, np.array(data_ch.tensor.shape))
            data_restored[:, :, ch] = ress.tensor
        data_restoredRGB = cv2.cvtColor(data_restored, cv2.COLOR_YCR_CB2RGB)
        iio.imwrite(oo, np.clip(data_restoredRGB, 0, 255).astype(np.uint8))
        iio.imwrite('diff.png', np.clip(np.abs(data_restoredRGB - data), 0, 255).astype(np.uint8))
        psnr_res = psnr(data.astype(np.float64), data_restoredRGB.astype(np.float64))
        if psnr_res < 300:
            print(f"Low PSNR! Content:{content_idx}, PSNR={psnr_res}")
        print('success')
        input()
