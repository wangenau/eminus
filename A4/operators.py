import numpy as np
from numpy.linalg import eig, det, inv
from numpy.fft import ifftn, fftn
from scipy.linalg import sqrtm
from setup import *

def O(inp):
    return det(R) * inp

def L(inp):
    inp = inp.T
    if inp.shape[1] == len(G2c):
        return (-det(R) * G2c * inp).T
    else:
        return (-det(R) * G2 * inp).T

def Linv(inp):
    inp = inp.T
    out = np.zeros(inp.shape, dtype=complex)
    if inp.ndim == 1:
        out[1:] = inp[1:] / G2[1:] / -det(R)
    else:
        for i in range(len(inp)):
            out[i][1:] = inp[i][1:] / G2[1:] / -det(R)
    return out.T

def K(inp):
    inp = inp.T
    out = np.empty(inp.shape, dtype=complex)
    if inp.shape[1] == len(G2c):
        if inp.ndim == 1:
            out = inp / (1 + G2c)
        else:
            for i in range(len(inp)):
                out[i] = inp[i] / (1 + G2c)
    else:
        if inp.ndim == 1:
            out = inp / (1 + G2)
        else:
            for i in range(len(inp)):
                out[i] = inp[i] / (1 + G2)
    return out.T

def cI(inp):
    inp = inp.T
    if inp.ndim == 1:
        inp = np.array([inp])
    if np.size(inp, 1) == np.prod(S):
        out = np.empty(inp.shape, dtype=complex)
        for i in range(inp.shape[0]):
            tmp = np.reshape(inp[i], S, order='F')
            out[i] = ifftn(tmp).flatten(order='F')
    else:
        out = np.empty((inp.shape[0], np.prod(S)), dtype=complex)
        for i in range(inp.shape[0]):
            full = np.zeros(np.prod(S), dtype=complex)
            full[active] = inp[i]
            full = np.reshape(full, S, order='F')
            out[i] = ifftn(full).flatten(order='F')
    return (out * np.prod(S)).T

def cJ(inp):
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, S, order='F')
        out = fftn(tmp).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(inp.shape[0]):
            tmp = np.reshape(inp[i], S, order='F')
            out[i] = fftn(tmp).flatten(order='F')
    return (out / np.prod(S)).T

def cIdag(inp):
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, S, order='F')
        full = fftn(tmp).flatten(order='F')
        out = full[active]
    else:
        out = np.empty((np.size(inp, 0), len(active[0])), dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], S, order='F')
            full = fftn(tmp).flatten(order='F')
            out[i] = full[active]
    return out.T

def cJdag(inp):
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, S, order='F')
        out = ifftn(tmp).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], S, order='F')
            out[i] = ifftn(tmp).flatten(order='F')
    return out.T

def diagouter(A, B):
    return np.sum(A * B.conj(), axis=1)

def getE(W, Vdual):
    Y = orth(W)
    n = getn(Y)
    phi = -4 * np.pi * Linv(O(cJ(n)))
    U = W.conj().T @ O(W)
    exc = excVWN(n)
    return np.real(-0.5 * np.trace(np.diag(f) @ (Y.conj().T @ L(Y))) + Vdual.conj().T @ n + \
           0.5 * n.conj().T @ cJdag(O(phi)) + n.conj().T @ cJdag(O(cJ(exc))))

def Diagprod(a, B):
    B = B.T
    return (a * B).T

def H(W, Vdual):
    Y = orth(W)
    n = getn(Y)
    phi = -4 * np.pi * Linv(O(cJ(n)))
    exc = excVWN(n)
    excp = excpVWN(n)
    Veff = Vdual + cJdag(O(phi)) + cJdag(O(cJ(exc))) + excp * cJdag(O(cJ(n)))
    return -0.5 * L(W) + cIdag(Diagprod(Veff, cI(W)))

def getgrad(W, Vdual):
    U = W.conj().T @ O(W)
    invU = inv(U)
    HW = H(W, Vdual)
    F = np.diag(f)
    U12 = sqrtm(inv(U))
    Ht = U12 @ (W.conj().T @ HW) @ U12
    return (HW - (O(W) @ invU) @ (W.conj().T @ HW)) @ (U12 @ F @ U12) + O(W) @ Q(Ht @ F - F @ Ht, U)

def sd(W, Vdual, Nit):
    Elist = []
    alpha = 3e-5
    for i in range(Nit):
        W = W - alpha * getgrad(W, Vdual)
        E = getE(W, Vdual)
        Elist.append(E)
        print(f'Nit: {i}  \tE(W): {E}')
    return W, np.asarray(Elist)

def lm(W, Vdual, Nit):
    Elist = []
    alphat = 3e-5
    g = getgrad(W, Vdual)
    d = -g
    gt = getgrad(W + alphat * d, Vdual)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = getE(W, Vdual)
    Elist.append(E)
    print(f'Nit: 0  \tE(W): {E}')
    for i in range(1, Nit):
        g = getgrad(W, Vdual)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -g
        gt = getgrad(W + alphat * d, Vdual)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = getE(W, Vdual)
        Elist.append(E)
        print(f'Nit: {i}  \tE(W): {E}  \tlinmin test: {linmin}')
    return W, np.asarray(Elist)

def pclm(W, Vdual, Nit):
    Elist = []
    alphat = 3e-5
    g = getgrad(W, Vdual)
    d = -K(g)
    gt = getgrad(W + alphat * d, Vdual)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = getE(W, Vdual)
    Elist.append(E)
    print(f'Nit: 0  \tE(W): {E}')
    for i in range(1, Nit):
        g = getgrad(W, Vdual)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -K(g)
        gt = getgrad(W + alphat * d, Vdual)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = getE(W, Vdual)
        Elist.append(E)
        print(f'Nit: {i}  \tE(W): {E}  \tlinmin test: {linmin}')
    return W, np.asarray(Elist)

def pccg(W, Vdual, Nit, cgform):
    Elist = []
    alphat = 3e-5
    g = getgrad(W, Vdual)
    d = -K(g)
    gt = getgrad(W + alphat * d, Vdual)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    dold = d
    gold = g
    E = getE(W, Vdual)
    Elist.append(E)
    print(f'Nit: 0  \tE(W): {E}')
    for i in range(1, Nit):
        g = getgrad(W, Vdual)
        linmin = dotprod(g, dold) / np.sqrt(dotprod(g, g) * dotprod(dold, dold))
        cg = dotprod(g, K(gold)) / np.sqrt(dotprod(g, K(g)) * dotprod(gold, K(gold)))
        if cgform == 1:
            beta = dotprod(g, K(g)) / dotprod(gold, K(gold))
        elif cgform == 2:
            beta = dotprod(g - gold, K(g)) / dotprod(gold, K(gold))
        elif cgform == 3:
            beta = dotprod(g - gold, K(g)) / dotprod(g - gold, dold)
        d = -K(g) + beta * dold
        gt = getgrad(W + alphat * d, Vdual)
        if abs(dotprod(g - gt, d)) < 1e-15:
            break
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        dold = d
        gold = g
        E = getE(W, Vdual)
        Elist.append(E)
        print(f'Nit: {i}  \tE(W): {E}  \tlinmin test: {linmin}  \tcg test: {cg}')
    return W, np.asarray(Elist)

def dotprod(a, b):
    return np.real(np.trace(a.conj().T @ b))

def orth(W):
    return W @ inv(sqrtm(W.conj().T @ O(W)))

def getPsi(W, Vdual):
    Y = orth(W)
    mu = Y.conj().T @ H(Y, Vdual)
    epsilon, D = eig(mu)
    return Y @ D, np.real(epsilon)

def excVWN(n):
    X1 = 0.75 * (3 / (2 * np.pi))**(2 / 3)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c
    rs = (4 * np.pi / 3 * n)**(-1/3)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    out = -X1 / rs + A * (np.log(x * x / X) + 2 * b / Q * np.arctan(Q / (2 * x + b)) \
    - (b * x0) / X0 * (np.log((x - x0) * (x - x0) / X) + 2 * (2 * x0 + b) / Q * np.arctan(Q / (2 * x + b))))
    return out

def excpVWN(n):
    X1 = 0.75 * (3 / (2 * np.pi))**(2 / 3)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c
    rs = (4 * np.pi / 3 * n)**(-1/3)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    dx = 0.5 / x
    out = dx * (2 * X1 / (rs * x) + A * (2 / x - (2 * x + b) / X - 4 * b / (Q * Q + (2 * x + b) * (2 * x + b)) \
    - (b * x0) / X0 * (2 / (x - x0) - (2 * x + b) / X - 4 * (2 * x0 + b) / (Q * Q + (2 * x + b) * (2 * x + b)))))
    out = (-rs / (3 * n)) * out
    return out

def getn(W):
    W = W.T
    n = np.zeros((np.prod(S), 1))
    for i in range(W.shape[0]):
        psi = cI(W[i])
        n += f[i] * np.real(psi.conj() * psi)
    return n.T[0]

def Q(inp, U):
    mu, V = eig(U)
    mu = np.reshape(mu, (len(mu), 1))
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T
