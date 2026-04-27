import argparse
import json
import os
from copy import deepcopy
from itertools import chain
from multiprocessing import Pool

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pywt

from batched import waverec_period_batched, wavedec_period_batched
from db import createWaveletFromContent, PRP_check
from denoise import universal_thresh, apply_bayes_thresh, apply_bayes_thresh_1d, universal_thresh_1d
from metrics import psnr
from noisegen import gen_gaussian_noise, gen_snp_noise
from offset_tensor import OffsetTensor
from periodic.wave import wavedec_period_fastest, waverec_period_fastest
from quant import apply_threshold_quantile, apply_threshold, apply_soft_threshold
from roundtrip import roundtrip
from skimage.metrics import structural_similarity as ssim

from utils import ci_size, decide_class


def benchmark(content):
    os.makedirs("results/{}".format(content["Index"]), exist_ok=True)
    row = dict()
    w = createWaveletFromContent(content)
    results = []
    if len(w.g) > np.rint(w.m).astype(int) - 1:
        return []
    print(len(w.g))
    #test_files = os.listdir('test')
    for file in ('lenna.bmp',):
        path = os.path.join('test', file)
        data = iio.imread(path)

        if data.ndim == 3:  # rgb
            data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
            #data = data.mean(axis=2)
        ci_class = pywt.wavedec2(data, "bior4.4", level=1, mode='periodization')
        img_class = decide_class(ci_class)
        iio.imwrite("results/{}/{}.png".format(content["Index"], file.split('.')[0]), data.astype(np.uint8))
        data = OffsetTensor(data, np.array([0, 0]))

        for level in (13,):#range(1,6):

            ci = wavedec_period_batched(data, w, level)
            res_true = waverec_period_batched(ci, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-true-l{}.png".format(content["Index"],
                                                                  file.split('.')[0],
                                                                  level
                                                                  ),
                        np.clip(res_true.tensor, 0, 255).astype(np.uint8))
            for thresh_quantile in (0.95,):
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
                print(ci_size(ci))
                row['Estimated_CR'] = data.tensor.size * (1 - thresh_quantile) / (ci_size(ci))
                row['TestImg'] = file
                row['ImgClass'] = img_class
                quantized, threshold = apply_threshold_quantile(ci, thresh_quantile)
                res = waverec_period_batched(quantized, w, np.array(data.tensor.shape))
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


def benchmark_denoise(content):
    os.makedirs("results/{}".format(content["Index"]), exist_ok=True)
    row = dict()
    w = createWaveletFromContent(content)
    results = []
    print(len(w.g))
    for file in ('tank2.bmp',):
        path = os.path.join('test', file)
        data = iio.imread(path)

        if data.ndim == 3:  # rgb
            data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114

        ci_class = pywt.wavedec2(data, "bior4.4", level=1, mode ='periodization')
        img_class = decide_class(ci_class)

        data_gaussian = gen_gaussian_noise(data)
        data_snp = gen_snp_noise(data)
        iio.imwrite("results/{}/{}.png".format(content["Index"], file.split('.')[0]), data.astype(np.uint8))
        data = OffsetTensor(data, np.array([0, 0]))
        data_gaussian = OffsetTensor(data_gaussian, np.array([0,0]))
        data_snp = OffsetTensor(data_snp, np.array([0,0]))

        for level in (4,):  # range(1,6):
            ci = wavedec_period_batched(data, w, level)
            ci_gaussian = wavedec_period_batched(data_gaussian, w, level)
            ci_snp = wavedec_period_batched(data_snp, w, level)
            res_true = waverec_period_batched(ci, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-true-l{}.png".format(content["Index"],
                                                            file.split('.')[0],
                                                            level
                                                            ),
                        np.clip(res_true.tensor, 0, 255).astype(np.uint8))
            res_gaussian = waverec_period_batched(ci_gaussian, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-gaussian-l{}.png".format(content["Index"],
                                                            file.split('.')[0],
                                                            level
                                                            ),
                        np.clip(res_gaussian.tensor, 0, 255).astype(np.uint8))
            res_snp = waverec_period_batched(ci_snp, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-snp-l{}.png".format(content["Index"],
                                                                file.split('.')[0],
                                                                level
                                                                ),
                        np.clip(res_snp.tensor, 0, 255).astype(np.uint8))
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
            row['ImgClass'] = img_class
            ci_gaussian_bayes = apply_bayes_thresh(ci_gaussian)
            ci_snp_bayes = apply_bayes_thresh(ci_snp)
            row['TestImg'] = file
            res_gaussian_bayes = waverec_period_batched(ci_gaussian_bayes, w, np.array(data.tensor.shape))
            res_snp_bayes = waverec_period_batched(ci_snp_bayes, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-l{}-{}.png".format(content["Index"],
                                                           file.split('.')[0],
                                                           level,
                                                           "gaussian-bayes"),
                        np.clip(res_gaussian_bayes.tensor, 0, 255).astype(np.uint8))
            psnr_gaussian_bayes = psnr(data.tensor, res_gaussian_bayes.tensor)
            ssim_gaussian_bayes = ssim(data.tensor, res_gaussian_bayes.tensor, data_range=256)
            row['PSNR_gaussian_bayes'] = psnr_gaussian_bayes
            row['SSIM_gaussian_bayes'] = ssim_gaussian_bayes
            iio.imwrite("results/{}/{}-l{}-{}.png".format(content["Index"],
                                                          file.split('.')[0],
                                                          level,
                                                          "snp-bayes"),
                        np.clip(res_snp_bayes.tensor, 0, 255).astype(np.uint8))
            psnr_snp_bayes = psnr(data.tensor, res_snp_bayes.tensor)
            ssim_snp_bayes = ssim(data.tensor, res_snp_bayes.tensor, data_range=256)
            row['PSNR_snp_bayes'] = psnr_snp_bayes
            row['SSIM_snp_bayes'] = ssim_snp_bayes
            thresh_visu_gaussian = universal_thresh(data_gaussian, ci)
            thresh_visu_snp = universal_thresh(data_snp, ci)
            row['TestImg'] = file
            ci_gaussian_visu, threshold = apply_soft_threshold(ci_gaussian, thresh_visu_gaussian)
            ci_snp_visu, threshold = apply_soft_threshold(ci_snp, thresh_visu_snp)
            res_gaussian_visu = waverec_period_batched(ci_gaussian_visu, w, np.array(data.tensor.shape))
            res_snp_visu = waverec_period_batched(ci_snp_visu, w, np.array(data.tensor.shape))
            iio.imwrite("results/{}/{}-l{}-{}.png".format(content["Index"],
                                                           file.split('.')[0],
                                                           level,
                                                           "gaussian-visu"),
                        np.clip(res_gaussian_visu.tensor, 0, 255).astype(np.uint8))
            iio.imwrite("results/{}/{}-l{}-{}.png".format(content["Index"],
                                                          file.split('.')[0],
                                                          level,
                                                          "snp-visu"),
                        np.clip(res_snp_visu.tensor, 0, 255).astype(np.uint8))
            psnr_gaussian_visu = psnr(data.tensor, res_gaussian_visu.tensor)
            ssim_gaussian_visu = ssim(data.tensor, res_gaussian_visu.tensor, data_range=256)
            psnr_snp_visu = psnr(data.tensor, res_snp_visu.tensor)
            ssim_snp_visu = ssim(data.tensor, res_snp_visu.tensor, data_range=256)
            row['PSNR_gaussian_visu'] = psnr_gaussian_visu
            row['SSIM_gaussian_visu'] = ssim_gaussian_visu
            row['PSNR_snp_visu'] = psnr_snp_visu
            row['SSIM_snp_visu'] = ssim_snp_visu

            results.append(deepcopy(row))
            print('appended')
    return results


def benchmark1D(wavename):
    results = []
    test_files = os.listdir('test')
    for file in test_files:
        path = os.path.join('test', file)
        data = iio.imread(path)
        if data.ndim == 3:  # rgb
            data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
            #data = data.mean(axis=2)
        for level in (2,3,):#range(1,6):
            coeffs = pywt.wavedec2(data, wavename, level= level, mode = 'periodization')
            lf_coef = coeffs[0]
            coeffs[0] = np.full_like(coeffs[0], 5000)
            arr, coeff_slices = pywt.coeffs_to_array(coeffs)

            for thresh_quantile in [0.975]:
                row = dict()
                row['Wavename'] = wavename
                row['Level'] = level
                row['Estimated_CR'] = thresh_quantile
                row['TestImg'] = file


                threshold = np.quantile(np.abs(arr.ravel()), thresh_quantile)
                arr_thresholded = pywt.threshold(arr, threshold, mode='hard')

                coeffs_from_arr = pywt.array_to_coeffs(arr_thresholded, coeff_slices, output_format='wavedec2')
                coeffs_from_arr[0] = lf_coef
                compressed_img = pywt.waverec2(coeffs_from_arr, wavename, mode = 'periodization')
                slices = tuple(slice(0, s) for s in data.shape)

                compressed_img = compressed_img[slices]
                psnr_uniform = psnr(data, compressed_img)
                ssim_uniform = ssim(data, compressed_img, data_range=256)
                row['PSNR_Uniform'] = psnr_uniform
                row['SSIM_Uniform'] = ssim_uniform
                results.append(row)

    return results

def benchmark1D_denoise(wavename):
    os.makedirs("results/{}".format(wavename), exist_ok=True)
    results = []
    test_files = os.listdir('test')
    for file in test_files:
        path = os.path.join('test', file)
        data = iio.imread(path)
        if data.ndim == 3:  # rgb
            data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
            #data = data.mean(axis=2)

        data_gaussian = gen_gaussian_noise(data)
        data_snp = gen_snp_noise(data)
        for level in (10,):#range(1,6):
            coeffs_gaussian = pywt.wavedecn(data_gaussian, wavename, level=level, mode='periodization')
            coeffs_snp = pywt.wavedecn(data_snp, wavename, level=level, mode='periodization')

            row = dict()
            row['Index'] = wavename
            row['Level'] = level
            row['TestImg'] = file

            ci_gaussian_bayes = apply_bayes_thresh_1d(coeffs_gaussian)
            ci_snp_bayes = apply_bayes_thresh_1d(coeffs_snp)

            img_gaussian_bayes = pywt.waverecn(ci_gaussian_bayes, wavename, mode = 'periodization')
            img_snp_bayes = pywt.waverecn(ci_snp_bayes, wavename, mode='periodization')
            slices = tuple(slice(0, s) for s in data.shape)

            img_gaussian_bayes = img_gaussian_bayes[slices]
            img_snp_bayes = img_snp_bayes[slices]
            iio.imwrite("results/{}/{}-l{}-{}.png".format(wavename,
                                                          file.split('.')[0],
                                                          level,
                                                          "gaussian-bayes"),
                        np.clip(img_gaussian_bayes, 0, 255).astype(np.uint8))
            iio.imwrite("results/{}/{}-l{}-{}.png".format(wavename,
                                                          file.split('.')[0],
                                                          level,
                                                          "snp-bayes"),
                        np.clip(img_snp_bayes, 0, 255).astype(np.uint8))
            psnr_gaussian_bayes = psnr(data, img_gaussian_bayes)
            ssim_gaussian_bayes = ssim(data, img_gaussian_bayes, data_range=256)
            psnr_snp_bayes = psnr(data, img_snp_bayes)
            ssim_snp_bayes = ssim(data, img_snp_bayes, data_range=256)
            row['PSNR_gaussian_bayes'] = psnr_gaussian_bayes
            row['SSIM_gaussian_bayes'] = ssim_gaussian_bayes
            row['PSNR_snp_bayes'] = psnr_snp_bayes
            row['SSIM_snp_bayes'] = ssim_snp_bayes

            thresh_gaussian_visu = universal_thresh_1d(data_gaussian, coeffs_gaussian)
            thresh_snp_visu = universal_thresh_1d(data_snp, coeffs_snp)

            coeffs_gaussian_visu = [coeffs_gaussian[0]] + [
                {
                    key: pywt.threshold(level[key], value=thresh_gaussian_visu, mode='soft')
                    for key in level
                }
                for level in coeffs_gaussian[1:]]
            coeffs_snp_visu = [coeffs_snp[0]] + [
                {
                    key: pywt.threshold(level[key], value=thresh_snp_visu, mode='soft')
                    for key in level
                }
                for level in coeffs_snp[1:]]

            img_gaussian_visu = pywt.waverecn(coeffs_gaussian_visu, wavename, mode='periodization')
            img_snp_visu = pywt.waverecn(coeffs_snp_visu, wavename, mode='periodization')
            img_gaussian_visu = img_gaussian_visu[slices]
            img_snp_visu = img_snp_visu[slices]
            iio.imwrite("results/{}/{}-l{}-{}.png".format(wavename,
                                                          file.split('.')[0],
                                                          level,
                                                          "gaussian-visu"),
                        np.clip(img_gaussian_visu, 0, 255).astype(np.uint8))
            iio.imwrite("results/{}/{}-l{}-{}.png".format(wavename,
                                                          file.split('.')[0],
                                                          level,
                                                          "snp-visu"),
                        np.clip(img_snp_visu, 0, 255).astype(np.uint8))
            psnr_gaussian_visu = psnr(data, img_gaussian_visu)
            ssim_gaussian_visu = ssim(data, img_gaussian_visu, data_range=256)
            psnr_snp_visu = psnr(data, img_snp_visu)
            ssim_snp_visu = ssim(data, img_snp_visu, data_range=256)
            row['PSNR_gaussian_visu'] = psnr_gaussian_visu
            row['SSIM_gaussian_visu'] = ssim_gaussian_visu
            row['PSNR_snp_visu'] = psnr_snp_visu
            row['SSIM_snp_visu'] = ssim_snp_visu
            print('appended')
            results.append(row)

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


        #with Pool(1) as p:
        results_nonflat = list(map(benchmark, contents))
        discrete_wavelets = pywt.wavelist(kind='discrete')
        results_1d = list(map(benchmark1D, discrete_wavelets))
        #results_nonflat = map(benchmark, contents[26:28])
        results = list(chain(*results_nonflat)) + list(chain(*results_1d))

        pd.DataFrame(results).to_csv('results.csv')
    elif args.command == 'benchmark_denoise':
        with open("WaveDB.json", 'r') as j:
            contents = json.loads(j.read())
        results_nonflat = list(map(benchmark_denoise, contents[10:11]))
        #discrete_wavelets = pywt.wavelist(kind='discrete')
        #results_1d = list(map(benchmark1D_denoise, discrete_wavelets))
        results = list(chain(*results_nonflat))# + list(chain(*results_1d))
        #results = list(chain(*results_1d))

        pd.DataFrame(results).to_csv('results-denoise.csv')



    else:
        print("Unknown command. Supported commands: roundtrip, benchmark, benchmark_denoise")
