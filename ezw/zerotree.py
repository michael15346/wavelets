import numpy as np
from bitarray import bitarray
import itertools

from ezw.wave import wavedec_ezw, waverec_ezw
from offset_tensor import OffsetTensor


def bytestuff(bits):
    marker = bitarray('11111111')
    zeros = bitarray('00000000')

    stuffed_arr = bitarray()

    idx = 0
    while idx < len(bits):
        cursor = bits[idx:idx + 8]
        stuffed_arr.extend(cursor)
        if cursor == marker:
            stuffed_arr.extend(zeros)
        idx += 8

    return stuffed_arr


PREFIX_FREE_CODE = {
    "T": bitarray('0'),
    "Z": bitarray('10'),
    "P": bitarray('110'),
    "N": bitarray('111')
}


class CoefficientTreeEZW:
    def __init__(self, value, level, hp_filter_index, loc, children=None):
        if children is None:
            children = []
        self.value = value
        self.level = level
        self.hp_filter_index = hp_filter_index
        self.children = children
        self.loc = loc
        self.code = None

    def zero_code(self, threshold):
        for child in self.children:
            child.zero_code(threshold)

        if abs(self.value) >= threshold:
            self.code = "P" if self.value > 0 else "N"
        else:
            self.code = "Z" if any([child.code != "T" for child in self.children]) else "T"

    @staticmethod
    def build_trees(coeffs, w):
        def build_children(level, loc, hp_filter_index):
            if level + 1 > len(coeffs): return []

            m = round(np.abs(np.linalg.det(w.M)))

            child_locs = list(range(m * loc, m * (loc + 1)))
            children = []
            for cloc in child_locs:
                if cloc >= coeffs[level][hp_filter_index].shape[0]:
                    continue
                node = CoefficientTreeEZW(coeffs[level][hp_filter_index][cloc], level, hp_filter_index, cloc)
                node.children = build_children(level + 1, cloc, hp_filter_index)
                children.append(node)
            return children

        LL = coeffs[0]

        LL_trees = []
        for i in range(LL.shape[0]):
            children = [CoefficientTreeEZW(subband[i], 1, hpf_index, i, children=build_children(2, i, hpf_index))
                        for hpf_index, subband in enumerate(coeffs[1])]

            LL_trees.append(CoefficientTreeEZW(LL[i], 0, None, i, children=children))

        return LL_trees


class ZeroTreeScan:
    def __init__(self, code, isDominant):
        self.isDominant = isDominant
        self.code = code
        self.bits = code if not isDominant else self.code_bits(code)

    def __len__(self):
        return len(self.bits)

    def tofile(self, file, padto=16):
        bits = self.bits.copy()

        if padto != 0 and len(bits) % padto != 0:
            bits.extend([False for _ in range(padto - (len(bits) % padto))])

        bits = bytestuff(bits)
        bits.tofile(file)

    def code_bits(self, code):
        bitarr = bitarray()
        bitarr.encode(PREFIX_FREE_CODE, code)
        return bitarr

    @staticmethod
    def from_bits(bits, isDominant):
        code = list(bits.decode(PREFIX_FREE_CODE)) if isDominant else bits
        return ZeroTreeScan(code, isDominant)


class ZeroTreeEncoderEZW:
    def __init__(self, wavedec_coeffs, w):
        flat_coef = np.array(list(wavedec_coeffs[0]) + list(itertools.chain(*itertools.chain(*wavedec_coeffs[1:]))))
        self.thresh = np.power(2, np.floor(np.log2(np.max(np.abs(flat_coef)))))

        # в оригинале было округление до ближайшего целого в сторону от нуля (в коэффициентах не будет нулей!)
        # надо ли?
        # wavedec_coeffs = np.sign(wavedec_coeffs) * np.floor(np.abs(wavedec_coeffs))

        self.trees = CoefficientTreeEZW.build_trees(wavedec_coeffs, w)

        self.start_thresh = self.thresh

        self.secondary_list = []
        self.perform_dominant_pass = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.thresh <= 0: raise StopIteration
        if self.thresh <= 1 and not self.perform_dominant_pass: raise StopIteration

        if self.perform_dominant_pass:
            scan, next_coeffs = self.dominant_pass()

            self.secondary_list = np.concatenate((self.secondary_list, next_coeffs))

            self.perform_dominant_pass = False
            return scan
        else:
            scan = self.secondary_pass()
            self.thresh //= 2
            self.perform_dominant_pass = True
            return scan

    def dominant_pass(self):
        sec = []

        q = []
        for parent in self.trees:
            parent.zero_code(self.thresh)
            q.append(parent)

        codes = []
        while len(q) != 0:
            node = q.pop(0)
            codes.append(node.code)

            if node.code != "T":
                for child in node.children:
                    q.append(child)

            if node.code == "P" or node.code == "N":
                sec.append(node.value)
                node.value = 0

        return ZeroTreeScan(codes, True), np.abs(np.array(sec))

    def secondary_pass(self):
        bits = bitarray()

        middle = self.thresh // 2
        for i, coeff in enumerate(self.secondary_list):
            if coeff - self.thresh >= 0:
                self.secondary_list[i] -= self.thresh
            bits.append(bool(self.secondary_list[i] >= middle))

        return ZeroTreeScan(bits, False)


class ZeroTreeDecoderEZW:
    def __init__(self, shape, start_thres, wavelet, level):
        img = OffsetTensor(np.zeros(shape), np.zeros_like(shape))
        self.shape = shape
        self.wavelet = wavelet
        self.coeffs = wavedec_ezw(img, wavelet, level) # might want to use dummy for this
        self.trees = CoefficientTreeEZW.build_trees(self.coeffs, wavelet)
        self.T = start_thres
        self.processed = []

    def getImage(self):
        return waverec_ezw(self.coeffs, self.wavelet, self.shape, np.zeros_like(self.shape))
        #return #pywt.waverec2(self.coeffs, self.wavelet)

    def process(self, scan):
        if scan.isDominant:
            self.dominant_pass(scan.code)
        else:
            self.secondary_pass(scan.code)

    def dominant_pass(self, code_list):
        q = []
        for parent in self.trees:
            q.append(parent)

        for code in code_list:
            if len(q) == 0:
                break
            node = q.pop(0)
            if code != "T":
                for child in node.children:
                    q.append(child)
            if code == "P" or code == "N":
                node.value = (1 if code == "P" else -1) * self.T
                self._fill_coeff(node)
                self.processed.append(node)

    def secondary_pass(self, bitarr):
        if len(bitarr) != len(self.processed):
            bitarr = bitarr[:len(self.processed)]
        for bit, node in zip(bitarr, self.processed):
            if bit:
                node.value += (1 if node.value > 0 else -1) * self.T // 2
                self._fill_coeff(node)

        self.T //= 2

    def _fill_coeff(self, node):
        if node.hp_filter_index is not None:
            self.coeffs[node.level][node.hp_filter_index][node.loc] = node.value
        else:
            self.coeffs[node.level][node.loc] = node.value