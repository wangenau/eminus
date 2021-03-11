#!/usr/bin/env python3
import numpy as np
from numpy.linalg import det, eig, inv, norm
from numpy.random import randn
from scipy.linalg import sqrtm
from utils import Diagprod, dotprod


def SCF(a, Nit_sd=10, Nit=100, cgform=1, Etol=1e-7):
    '''Main SCF function.'''
    W = randn(len(a.active[0]), a.Ns) + 1j * randn(len(a.active[0]), a.Ns)
    W = orth(a, W)
    W, Elist = sd(a, W, Nit_sd, Etol)
    W = orth(a, W)
    W, Elist = pccg(a, W, Nit, Etol, cgform)
    Eel = Elist[-1]
    EEwald = getEwald(a)
    Etot = Eel + EEwald
    if a.verbose >= 4:
        print('Compression: %f'% (len(a.G2) / len(a.G2c)))
    if a.verbose >= 3:
        print(f'Ewald energy: {EEwald} Eh')
        print(f'Electronic energy: {Eel} Eh')
    print(f'Total energy: {Etot} Eh')
    a.psi, a.epsilon = getPsi(a, W)
    a.n = getn(a, W)
    a.etot = Etot
    return


def H(a, W):
    '''Left-hand side of our eigenvalue equation.'''
    W = orth(a, W)
    n = getn(a, W)
    phi = -4 * np.pi * a.Linv(a.O(a.J(n)))
    exc = excVWN(n)
    excp = excpVWN(n)
    Veff = a.Vdual + a.Jdag(a.O(phi)) + a.Jdag(a.O(a.J(exc))) + excp * a.Jdag(a.O(a.J(n)))
    return -0.5 * a.L(W) + a.Idag(Diagprod(Veff, a.I(W)))


def getE(a, W):
    '''Calculate the sum of energies over Ns states.'''
    W = orth(a, W)
    n = getn(a, W)
    phi = -4 * np.pi * a.Linv(a.O(a.J(n)))
    U = W.conj().T @ a.O(W)
    exc = excVWN(n)
    return np.real(-0.5 * np.trace(np.diag(a.f) @ (W.conj().T @ a.L(W))) + a.Vdual.conj().T @ n + \
           0.5 * n.conj().T @ a.Jdag(a.O(phi)) + n.conj().T @ a.Jdag(a.O(a.J(exc))))


def getEwald(a):
    '''Calculate the Ewald/Coulomb energy.'''
    dr = norm(a.r - np.sum(a.R, axis=1) / 2, axis=1)
    sigma1 = 0.25
    g1 = np.exp(-dr**2 / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)**3
    g1 = a.Z * (np.sum(g1) * det(a.R) / np.prod(a.S)) * g1
    n = a.I(a.J(g1) * a.Sf)
    n = np.real(n)
    phi = a.I(a.Linv(-4 * np.pi * a.O(a.J(n))))
    phi = np.real(phi)
    Unum = 0.5 * np.real(a.J(phi).conj().T @ a.O(a.J(n)))
    Uself = a.Z**2 / (2 * np.sqrt(np.pi)) * (1 / sigma1) * a.X.shape[0]
    return (Unum - Uself)[0][0]


def getgrad(a, W):
    '''Calculate the energy gradient with respect to W.'''
    U = W.conj().T @ a.O(W)
    invU = inv(U)
    HW = H(a, W)
    F = np.diag(a.f)
    U12 = sqrtm(inv(U))
    Ht = U12 @ (W.conj().T @ HW) @ U12
    return (HW - (a.O(W) @ invU) @ (W.conj().T @ HW)) @ (U12 @ F @ U12) + a.O(W) @ Q(Ht @ F - F @ Ht, U)


def getPsi(a, W):
    '''Calculate eigensolutions and eigenvalues from the coefficent matrix W.'''
    W = orth(a, W)
    mu = W.conj().T @ H(a, W)
    epsilon, D = eig(mu)
    return W @ D, np.real(epsilon)


def getn(a, W):
    '''Generate the electronic density.'''
    W = W.T
    n = np.zeros((np.prod(a.S), 1))
    for i in range(W.shape[0]):
        psi = a.I(W[i])
        n += a.f[i] * np.real(psi.conj() * psi)
    return n.T[0]


def sd(a, W, Nit, Etol):
    '''Steepest descent minimization algorithm.'''
    print('Start steepest descent minimization...')
    Elist = []
    alpha = 3e-5
    for i in range(Nit):
        W = W - alpha * getgrad(a, W)
        E = getE(a, W)
        Elist.append(E)
        if a.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E}')
        if i > 0:
            if abs(Elist[-2] - Elist[-1]) < Etol:
                break
    return W, np.asarray(Elist)


def lm(a, W, Nit, Etol):
    '''Line minimization algorithm.'''
    print('Start Line minimization...')
    Elist = []
    alphat = 3e-5
    g = getgrad(a, W)
    d = -g
    gt = getgrad(a, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = getE(a, W)
    Elist.append(E)
    if a.verbose >= 3:
        print(f'Nit: 1  \tE(W): {E}')
    for i in range(1, Nit):
        g = getgrad(a, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -g
        gt = getgrad(a, W + alphat * d)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = getE(a, W)
        Elist.append(E)
        if a.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E}  \tlinmin test: {linmin}')
        if abs(Elist[-2] - Elist[-1]) < Etol:
            break
    return W, np.asarray(Elist)


def pclm(a, W, Nit, Etol):
    '''Preconditioned line minimization algorithm.'''
    print('Start preconditioned line minimization...')
    Elist = []
    alphat = 3e-5
    g = getgrad(a, W)
    d = -a.K(g)
    gt = getgrad(a, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = getE(a, W)
    Elist.append(E)
    if a.verbose >= 3:
        print(f'Nit: 1  \tE(W): {E}')
    for i in range(1, Nit):
        g = getgrad(a, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -a.K(g)
        gt = getgrad(a, W + alphat * d)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = getE(a, W)
        Elist.append(E)
        if a.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E}  \tlinmin test: {linmin}')
        if abs(Elist[-2] - Elist[-1]) < Etol:
            break
    return W, np.asarray(Elist)


def pccg(a, W, Nit, Etol, cgform=1):
    '''Preconditioned conjugate-gradient algorithm.'''
    print('Start preconditioned conjugate-gradient minimization...')
    Elist = []
    alphat = 3e-5
    g = getgrad(a, W)
    d = -a.K(g)
    gt = getgrad(a, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    dold = d
    gold = g
    E = getE(a, W)
    Elist.append(E)
    if a.verbose >= 3:
        print(f'Nit: 0  \tE(W): {E}')
    for i in range(1, Nit):
        g = getgrad(a, W)
        linmin = dotprod(g, dold) / np.sqrt(dotprod(g, g) * dotprod(dold, dold))
        cg = dotprod(g, a.K(gold)) / np.sqrt(dotprod(g, a.K(g)) * dotprod(gold, a.K(gold)))
        if cgform == 1:
            beta = dotprod(g, a.K(g)) / dotprod(gold, a.K(gold))
        elif cgform == 2:
            beta = dotprod(g - gold, a.K(g)) / dotprod(gold, a.K(gold))
        elif cgform == 3:
            beta = dotprod(g - gold, a.K(g)) / dotprod(g - gold, dold)
        d = -a.K(g) + beta * dold
        gt = getgrad(a, W + alphat * d)
        # If this becomes zero, the result will become nonsense
        if abs(dotprod(g - gt, d)) < 1e-16:
            break
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        dold = d
        gold = g
        E = getE(a, W)
        Elist.append(E)
        if a.verbose >= 3:
            print(f'Nit: {i}  \tE(W): {E}  \tlinmin test: {linmin}  \tcg test: {cg}')
        if abs(Elist[-2] - Elist[-1]) < Etol:
            break
    return W, np.asarray(Elist)


def excVWN(n):
    '''VWN parameterization of the exchange correlation energy function.'''
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
    '''Derivation with respect to n of the VWN parameterization of the exchange correlation energy function.'''
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


def orth(a, W):
    '''Orthogonalize coefficent matrix W.'''
    return W @ inv(sqrtm(W.conj().T @ a.O(W)))


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.'''
    mu, V = eig(U)
    mu = np.reshape(mu, (len(mu), 1))
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T
