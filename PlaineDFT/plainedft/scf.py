#!/usr/bin/env python3
'''
Main SCF file with every relevant function.
'''
import numpy as np
from numpy.linalg import det, eig, inv, norm
from numpy.random import randn, seed
from scipy.linalg import sqrtm
from timeit import default_timer
from .utils import Diagprod, dotprod
from .gth_nonloc import calc_Enl, calc_Vnl


def SCF(a, n_sd=10, n_lm=0, n_pclm=0, n_cg=100, cgform=1, etol=1e-7):
    '''Main SCF function.'''
    # Set up basis functions
    # Start with randomized, complex, orthogonal basis functions
    # seed(1234)
    W = randn(len(a.active[0]), a.Ns) + 1j * randn(len(a.active[0]), a.Ns)
    W = orth(a, W)

    # Minimization procedure
    start = default_timer()
    if n_sd > 0:
        W, Elist = sd(a, W, n_sd, etol)
    if n_lm > 0:
        W, Elist = lm(a, W, n_lm, etol)
    if n_pclm > 0:
        W, Elist = pclm(a, W, n_pclm, etol)
    if n_cg > 0:
        W, Elist = pccg(a, W, n_cg, etol, cgform)
    end = default_timer()

    # Handle energies and output
    Eel = Elist[-1]
    EEwald = getEwald(a)
    Etot = Eel + EEwald
    if a.verbose >= 5:
        print(f'Compression: {len(a.G2) / len(a.G2c):.5f}')
    if a.verbose >= 4:
        print(f'Time spent: {end - start:.5f}s')
    if a.verbose >= 3:
        getE(a, W, True)
        print(f'Electronic energy:           {Eel:+.9f} Eh')
        print(f'Ewald energy:                {EEwald:+.9f} Eh')
    print(f'Total energy:                {Etot:+.9f} Eh')

    # Save calculation parameters
    a.psi, a.estate = getPsi(a, W)  # Save wave functions and
    a.n = getn(a, W)  # Save electronic density
    a.etot = Etot  # Save total energy
    return


def H(a, W):
    '''Left-hand side of our eigenvalue equation.'''
    W = orth(a, W)  # Orthogonalize at the start
    n = getn(a, W)
    phi = -4 * np.pi * a.Linv(a.O(a.J(n)))
    exc = excVWN(n)
    excp = excpVWN(n)
    Vdual = a.Vloc
    Veff = Vdual + a.Jdag(a.O(phi)) + a.Jdag(a.O(a.J(exc))) + excp * a.Jdag(a.O(a.J(n)))
    Vnlpsi = 0
    if a.NbetaNL > 0:  # Only calculate non-local potential if necessary
        Vnlpsi = calc_Vnl(a, W)
    return -0.5 * a.L(W) + a.Idag(Diagprod(Veff, a.I(W))) + Vnlpsi


def getE(a, W, out=False):
    '''Calculate the sum of energies over Ns states.'''
    W = orth(a, W)  # Orthogonalize at the start
    n = getn(a, W)
    phi = -4 * np.pi * a.Linv(a.O(a.J(n)))
    U = W.conj().T @ a.O(W)
    exc = excVWN(n)
    Ekin = np.real(-0.5 * np.trace(np.diag(a.f) @ (W.conj().T @ a.L(W))))
    Eloc = np.real(a.Vloc.conj().T @ n)
    Enonloc = 0
    if a.NbetaNL > 0:  # Only calculate non-local energy if necessary
        Enonloc = calc_Enl(a, W)
    Ecoul = np.real(0.5 * n.conj().T @ a.Jdag(a.O(phi)))
    Exc = np.real(n.conj().T @ a.Jdag(a.O(a.J(exc))))
    if a.verbose >= 5 or out:
        print(f'Kinetic energy:              {Ekin:+.9f} Eh')
        print(f'Local potential energy:      {Eloc:+.9f} Eh')
        print(f'Non-local potential energy:  {Enonloc:+.9f} Eh')
        print(f'Coulomb energy:              {Ecoul:+.9f} Eh')
        print(f'Exchange-correlation energy: {Exc:+.9f} Eh')
    return Ekin + Eloc + Enonloc + Ecoul + Exc

# FIXME: Dimensions are scuffed
def getEwald(a):
    '''Calculate the Ewald/Coulomb energy.'''
    dr = norm(a.r - np.sum(a.R, axis=1) / 2, axis=1)
    sigma1 = 0.25
    g1 = np.exp(-dr**2 / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)**3
    # FIXME: What about different Zs?
    g1 = a.Z[0] * (np.sum(g1) * det(a.R) / np.prod(a.S)) * g1
    n = a.I(a.J(g1) * a.Sf)
    n = np.real(n)
    phi = a.I(a.Linv(-4 * np.pi * a.O(a.J(n))))
    phi = np.real(phi)
    Unum = 0.5 * np.real(a.J(phi).conj().T @ a.O(a.J(n)))
    # FIXME: What about different Zs?
    Uself = a.Z[0]**2 / (2 * np.sqrt(np.pi)) * (1 / sigma1) * a.X.shape[0]
    return (Unum[0][0] - Uself)


def getgrad(a, W):
    '''Calculate the energy gradient with respect to W.'''
    U = W.conj().T @ a.O(W)
    invU = inv(U)
    HW = H(a, W)
    F = np.diag(a.f)
    U12 = sqrtm(inv(U))
    Ht = U12 @ (W.conj().T @ HW) @ U12
    return (HW - (a.O(W) @ invU) @ (W.conj().T @ HW)) @ (U12 @ F @ U12) + a.O(W) @ (U12 @ Q(Ht @ F - F @ Ht, U))


# def getgrad(a, W):
#     grad = np.zeros(W.size)
#     H_psi = H(a, W)
#     for ist in range(a.Ns):
#         grad[:, ist] = H_psi[:, ist]
#         for range(a.Ns):
#             grad[:, ist] -= np.dot(psi[:, jst], H_psi[:, ist]) * psi[:, jst]
#         grad[:,ist] *= a.f[ist]
#
#     if np.all(a.f == 2) or np.all(a.f == 1):
#         return grad
#
#     F = np.diag(a.f)
#     H = W.T * H_psi
#     HFH = H*F - F*H
#     Q = 0.5 * HFH
#     grad[:,:] += psi * H
#     return grad


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


def sd(a, W, Nit, etol):
    '''Steepest descent minimization algorithm.'''
    print('Start steepest descent minimization...')
    Elist = []
    alpha = 3e-5
    for i in range(Nit):
        W = W - alpha * getgrad(a, W)
        E = getE(a, W)
        Elist.append(E)
        if a.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E:+.7f} Eh')
        if i > 0:
            if abs(Elist[-2] - Elist[-1]) < etol:
                print(f'Converged after {i+1} steps.')
                break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def lm(a, W, Nit, etol):
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
        print(f'Nit: 1  \tE(W): {E:+.7f} Eh')
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
            print(f'Nit: {i+1}  \tE(W): {E:+.7f} Eh\tlinmin test: {linmin:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def pclm(a, W, Nit, etol):
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
        print(f'Nit: 1  \tE(W): {E:+.7f} Eh')
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
            print(f'Nit: {i+1}  \tE(W): {E:+.7f} Eh\tlinmin test: {linmin:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def pccg(a, W, Nit, etol, cgform=1):
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
        print(f'Nit: 0  \tE(W): {E:+.7f} Eh')
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
        # FIXME: This feels wrong
        # If this becomes zero, the result will become nonsense
        if abs(dotprod(g - gt, d)) == 0:
            break
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        dold = d
        gold = g
        E = getE(a, W)
        Elist.append(E)
        if a.verbose >= 3:
            print(f'Nit: {i}  \tE(W): {E:+.7f} Eh\tlinmin test: {linmin:+.7f}  \tcg test: {cg:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
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
    return (-rs / (3 * n)) * out


def orth(a, W):
    '''Orthogonalize coefficent matrix W.'''
    return W @ inv(sqrtm(W.conj().T @ a.O(W)))


# FIXME: Only for testing
# def orth2(a, W):
#     '''Orthogonalize coefficent matrix W.'''
#     return W @ inv(sqrtm(W.conj().T @ W))


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.'''
    mu, V = eig(U)
    mu = np.reshape(mu, (len(mu), 1))
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T
