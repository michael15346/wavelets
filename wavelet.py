from dataclasses import dataclass
import numpy as np
from offset_matrix import OffsetMatrix


@dataclass
class Wavelet:
    h: OffsetMatrix
    g: tuple[OffsetMatrix, ...]
    hdual: OffsetMatrix
    gdual: tuple[OffsetMatrix, ...]
    M: np.ndarray
    m: float

    def __init__(self, h, g, hdual, gdual, M, m):
        self.h = h
        self.g = g
        self.hdual = hdual
        self.gdual = gdual
        self.M = M
        self.m = m

