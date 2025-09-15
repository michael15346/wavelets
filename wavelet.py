from dataclasses import dataclass
import numpy as np
from offset_tensor import OffsetTensor


@dataclass
class Wavelet:
    h: OffsetTensor
    g: tuple[OffsetTensor, ...]
    hdual: OffsetTensor
    gdual: tuple[OffsetTensor, ...]
    M: np.ndarray
    m: float

    def __init__(self, h, g, hdual, gdual, M, m):
        self.h = h
        self.g = g
        self.hdual = hdual
        self.gdual = gdual
        self.M = M
        self.m = m

    def __str__(self):
        return f"""Wavelet(
h:
{self.h}
g:
{self.g}
hdual:
{self.hdual}
gdual:
{self.gdual}
M:
{self.M}
det(M):
{self.m}
"""
    def __repr__(self):
        return str(self)

