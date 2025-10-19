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
from quant import encode_kmeans, decode_kmeans, entropy, encode_uniform, decode_uniform
from roundtrip import roundtrip
from skimage.metrics import structural_similarity as ssim
def benchmark(content):
    os.makedirs("results/{}".format(content["Index"]), exist_ok=True)
    row = dict()
    w = createWaveletFromContent(content)
    print(w)
    results = []
    #test_files = os.listdir('test')
    for file in ('lenna.bmp',):
        path = os.path.join('test', file)
        data = iio.imread(path)

        if data.ndim == 3:  # rgb
            data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
            #data = data.mean(axis=2)
        iio.imwrite("results/{}/{}.png".format(content["Index"], file.split('.')[0]), data.astype(np.uint8))
        data = OffsetTensor(data, np.array([0, 0]))
        for level in (5,):
            ci = wavedec_ezw(data, w, level)
            res_true = waverec_ezw(ci, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-true-l{}.png".format(content["Index"],
                                                                  file.split('.')[0],
                                                                  level
                                                                  ),
                        np.clip(res_true.tensor, 0, 255).astype(np.uint8))
            for log_clusters in (10,):
                row['Index'] = content['Index']
                row['WaveletSystemType'] = content['WaveletSystemType']
                row['RefinableMaskInfo'] = content['RefinableMaskInfo']
                row['DualRefinableMaskInfo'] = content['DualRefinableMaskInfo']
                row['SymmetryInfo'] = content['SymmetryInfo']
                row['DilationMatrixInfo'] = content['DilationMatrixInfo']
                row['SR'] = content['Mask']['SR']
                row['DualSR'] = content['DualMask']['SR']
                row['ValidPRP'] = PRP_check(w)[0]
                row['SourceInfo'] = content['SourceInfo']
                row['Quant'] = 'KMeans'
                row['Level'] = level
                row['Clusters'] = 2 ** log_clusters
                row['TestImg'] = file
                centroids_kmeans, cluster_kmeans = encode_kmeans(ci, 2 ** log_clusters)
                entropy_kmeans = entropy(cluster_kmeans)
                row['Entropy_KMeans'] = entropy_kmeans

                ci_kmeans = decode_kmeans(centroids_kmeans, cluster_kmeans, w, data.tensor.shape, level)
                res_kmeans = waverec_ezw(ci_kmeans, w, np.array(data.tensor.shape))
                iio.imwrite("results/{}/{}-kmeans-l{}-c{}.png".format(content["Index"],
                                                                      file.split('.')[0],
                                                                      level,
                                                                      2 ** log_clusters
                                                                      ),
                            np.clip(res_kmeans.tensor, 0, 255).astype(np.uint8))
                psnr_kmeans = psnr(data.tensor, res_kmeans.tensor)
                ssim_kmeans = ssim(data.tensor, res_kmeans.tensor, data_range=256)
                row['PSNR_KMeans'] = psnr_kmeans
                row['SSIM_KMeans'] = ssim_kmeans
                quantized_uniform, n_min, n_max = encode_uniform(ci, 2 ** log_clusters)
                entropy_uniform = entropy(quantized_uniform)
                row['Entropy_Uniform'] = entropy_uniform
                ci_uniform = decode_uniform(quantized_uniform, n_min, n_max, 2 ** log_clusters, w, data.tensor.shape, level)
                res_uniform = waverec_period(ci_uniform, w, np.array(data.tensor.shape))
                iio.imwrite("results/{}/{}-uniform-l{}-c{}.png".format(content["Index"],
                                                                       file.split('.')[0],
                                                                       level,
                                                                       2 ** log_clusters),
                            np.clip(res_uniform.tensor, 0, 255).astype(np.uint8))
                psnr_uniform = psnr(data.tensor, res_uniform.tensor)
                ssim_uniform = ssim(data.tensor, res_uniform.tensor, data_range=256)
                row['PSNR_Uniform'] = psnr_uniform
                row['SSIM_Uniform'] = ssim_uniform
                results.append(deepcopy(row))
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


        with Pool(8) as p:
         results_nonflat = p.map(benchmark, contents)
        results = list(chain(*results_nonflat))

        pd.DataFrame(results).to_csv('results.csv')



    else:
        print("Unknown command. Supported commands: roundtrip, benchmark")
