import argparse
import json
import os
from copy import deepcopy
from itertools import chain
from multiprocessing import Pool

import imageio.v3 as iio
import numpy as np
import pandas as pd

from db import createWaveletFromContent, checkContent, PRP_check
from ezw.wave import waverec_ezw, wavedec_ezw
from metrics import psnr
from offset_tensor import OffsetTensor
from periodic.wave import waverec_period, wavedec_period
from quant import encode_kmeans, decode_kmeans, entropy, encode_uniform, decode_uniform, apply_threshold
from roundtrip import roundtrip
from skimage.metrics import structural_similarity as ssim
def benchmark(content):
    os.makedirs("results/{}".format(content["Index"]), exist_ok=True)
    row = dict()
    w = createWaveletFromContent(content)
    results = []
    if len(w.g) > np.rint(w.m).astype(int) - 1:
        return []
    #test_files = os.listdir('test')
    for file in ('lenna.bmp',):
        path = os.path.join('test', file)
        data = iio.imread(path)

        if data.ndim == 3:  # rgb
            data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
            #data = data.mean(axis=2)
        iio.imwrite("results/{}/{}.png".format(content["Index"], file.split('.')[0]), data.astype(np.uint8))
        data = OffsetTensor(data, np.array([0, 0]))

        
        if len(w.g) == 1:
            max_level = 9
        elif len(w.g) == 2:
            max_level = 6
        else:
            max_level = 6
        for level in (max_level,):#range(1,6):

            ci = wavedec_period(data, w, level)
            res_true = waverec_period(ci, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-true-l{}.png".format(content["Index"],
                                                                  file.split('.')[0],
                                                                  level
                                                                  ),
                        np.clip(res_true.tensor, 0, 255).astype(np.uint8))
            for thresh_quantile in np.logspace(-8,0,5, base=2):
                row['Index'] = content['Index']
                row['WaveletSystemType'] = content['WaveletSystemType']
                row['RefinableMaskInfo'] = content['RefinableMaskInfo']
                row['DualRefinableMaskInfo'] = content['DualRefinableMaskInfo']
                row['SymmetryInfo'] = content['SymmetryInfo']
                row['DilationMatrixInfo'] = content['DilationMatrixInfo']
                row['WaveletMaskAmount'] = len(w.g)
                row['SR'] = content['Mask']['SR']
                row['DualSR'] = content['DualMask']['SR']
                row['ValidPRP'] = PRP_check(w)[0]
                row['SourceInfo'] = content['SourceInfo']
                row['Level'] = level
                row['Estimated_CR'] = thresh_quantile
                row['TestImg'] = file
                quantized, threshold = apply_threshold(ci, thresh_quantile)
                #entropy_q = entropy(quantized)
                #row['Entropy'] = entropy_q
                res = waverec_period(quantized, w, np.array(data.tensor.shape))
                iio.imwrite("results/{}/{}-l{}-q{}.png".format(content["Index"],
                                                                       file.split('.')[0],
                                                                       level,
                                                                       thresh_quantile),
                            np.clip(res.tensor, 0, 255).astype(np.uint8))
                psnr_uniform = psnr(data.tensor, res.tensor)
                ssim_uniform = ssim(data.tensor, res.tensor, data_range=256)
                row['PSNR'] = psnr_uniform
                row['SSIM'] = ssim_uniform
                results.append(deepcopy(row))
                print('appended')
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='wavelets')
    parser.add_argument('command')
    parser.add_argument('-i')
    parser.add_argument('-o')
    args = parser.parse_args()
    if args.command == 'roundtrip':
        roundtrip(args.i, args.o)
    elif args.command == 'benchmark':
        with open("WaveDB.json", 'r') as j:
            contents = json.loads(j.read())
        #results_nonflat = []
        #results_nonflat.append(benchmark(contents[0]))
        #for c in contents:


        with Pool(4) as p:
            results_nonflat = list(map(benchmark, contents))
        #results_nonflat = map(benchmark, contents[26:28])
        results = list(chain(*results_nonflat))

        pd.DataFrame(results).to_csv('results.csv')



    else:
        print("Unknown command. Supported commands: roundtrip, benchmark")
