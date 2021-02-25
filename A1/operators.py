import numpy as np
from numpy.linalg import det, pinv
from numpy.fft import ifftn, fftn
from setup import *

def O(inp):
    return det(R) * inp

def L(inp):
    return -det(R) * G2 * inp

def Linv(inp):
    out = inp[1:] / G2[1:] / -det(R)
    return np.concatenate(([0], out))

def cI(inp):
    inp = np.reshape(inp, S, order='F')
    out = ifftn(inp).flatten(order='F')
    return out * np.prod(S)

def cJ(inp):
    inp = np.reshape(inp, S, order='F')
    out = fftn(inp).flatten(order='F')
    return out / np.prod(S)
