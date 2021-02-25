import numpy as np
from numpy.linalg import eig, det, inv
from numpy.fft import ifftn, fftn
from scipy.linalg import sqrtm
from setup import *

def O(inp):
    return det(R) * inp

def L(inp):
    inp = inp.T
    return (-det(R) * G2 * inp).T

def Linv(inp):
    out = np.zeros(inp.shape, dtype=complex)
    if inp.ndim == 1:
        out[1:] = inp[1:] / G2[1:] / -det(R)
    else:
        for i in range(len(inp)):
            out[i][1:] = inp[i][1:] / G2[1:] / -det(R)
    return out

def cI(inp):
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, S, order='F')
        out = ifftn(tmp).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], S, order='F')
            out[i] = ifftn(tmp).flatten(order='F')
    return (out * np.prod(S)).T

def cJ(inp):
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, S, order='F')
        out = fftn(tmp).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], S, order='F')
            out[i] = fftn(tmp).flatten(order='F')
    return (out / np.prod(S)).T

def cIdag(inp):
    return cJ(inp) * np.prod(S)

def cJdag(inp):
    return cI(inp) / np.prod(S)

def diagouter(A, B):
    #return np.diag(A @ B.conj().T)
    return np.sum(A * B.conj(), axis=1)

def getE(W, Vdual):
    U = W.conj().T @ O(W)
    invU = inv(U)
    cIW = cI(W)
    n = diagouter(cIW @ invU, cIW)
    return np.real(-0.5 * np.sum(diagouter(W.conj().T, L(W @ invU).conj().T)) + Vdual.conj().T @ n)

def Diagprod(a, B):
    B = B.T
    return (a * B).T

def H(W, Vdual):
    return -0.5 * L(W) + cIdag(Diagprod(Vdual, cI(W)))

def getgrad(W, Vdual):
    U = W.conj().T @ O(W)
    invU = inv(U)
    HW = H(W, Vdual)
    return (HW - (O(W) @ invU) @ (W.conj().T @ HW)) @ invU

def sd(W, Vdual, Nit):
    alpha = 3e-5
    for i in range(Nit):
        W = W - alpha * getgrad(W, Vdual)
        print(f'Nit: {i}  \tE(W): {getE(W, Vdual)}')
    return W

def orth(W):
    return W @ inv(sqrtm(W.conj().T @ O(W)))

def getPsi(W, Vdual):
    Y = orth(W)
    mu = Y.conj().T @ H(Y, Vdual)
    epsilon, D = eig(mu)
    return Y @ D, np.real(epsilon)
