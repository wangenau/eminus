#!/usr/bin/env python3
'''
Main SCF file with every relevant function.
'''
import numpy as np
from numpy.linalg import det, eig, inv, norm
from numpy.random import randn
from scipy.linalg import sqrtm
from timeit import default_timer
from .utils import Diagprod, dotprod
from .gth_nonloc import calc_Enonloc, calc_Vnonloc


def SCF(atoms, n_sd=10, n_lm=0, n_pclm=0, n_cg=100, cgform=1, etol=1e-7):
    '''Main SCF function.'''
    # Set up basis functions
    # Start with randomized, complex, orthogonal basis functions
    W = randn(len(atoms.active[0]), atoms.Ns) + 1j * randn(len(atoms.active[0]), atoms.Ns)
    W = orth(atoms, W)

    # Minimization procedure
    start = default_timer()
    if n_sd > 0:
        W, Elist = sd(atoms, W, n_sd, etol)
    if n_lm > 0:
        W, Elist = lm(atoms, W, n_lm, etol)
    if n_pclm > 0:
        W, Elist = pclm(atoms, W, n_pclm, etol)
    if n_cg > 0:
        W, Elist = pccg(atoms, W, n_cg, etol, cgform)
    end = default_timer()

    # Handle energies and output
    Eel = Elist[-1]
    EEwald = getEwald(atoms)
    Etot = Eel + EEwald
    if atoms.verbose >= 5:
        print(f'Compression: {len(atoms.G2) / len(atoms.G2c):.5f}')
    if atoms.verbose >= 4:
        print(f'Time spent: {end - start:.5f}s')
    if atoms.verbose >= 3:
        getE(atoms, W, True)
        print(f'Electronic energy:           {Eel:+.9f} Eh')
        print(f'Ewald energy:                {EEwald:+.9f} Eh')
    print(f'Total energy:                {Etot:+.9f} Eh')

    # Save calculation parameters
    atoms.psi, atoms.estate = getPsi(atoms, W)  # Save wave functions and
    atoms.n = getn(atoms, W)  # Save electronic density
    atoms.etot = Etot  # Save total energy
    return


def H(atoms, W):
    '''Left-hand side of our eigenvalue equation.'''
    W = orth(atoms, W)  # Orthogonalize at the start
    n = getn(atoms, W)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    exc = excVWN(n)
    excp = excpVWN(n)
    Vdual = atoms.Vloc
    Veff = Vdual + atoms.Jdag(atoms.O(phi)) + atoms.Jdag(atoms.O(atoms.J(exc))) + excp * atoms.Jdag(atoms.O(atoms.J(n)))
    Vnonloc_psi = 0
    if atoms.NbetaNL > 0:  # Only calculate non-local potential if necessary
        Vnonloc_psi = calc_Vnonloc(atoms, W)
    Vkin_psi = -0.5 * atoms.L(W)
    return Vkin_psi + atoms.Idag(Diagprod(Veff, atoms.I(W))) + Vnonloc_psi


def getE(atoms, W, out=False):
    '''Calculate the sum of energies over Ns states.'''
    W = orth(atoms, W)  # Orthogonalize at the start
    n = getn(atoms, W)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    exc = excVWN(n)
    Ekin = np.real(-0.5 * np.trace(np.diag(atoms.f) @ (W.conj().T @ atoms.L(W))))
    Eloc = np.real(atoms.Vloc.conj().T @ n)# * atoms.CellVol / np.prod(atoms.S)
    Enonloc = 0
    if atoms.NbetaNL > 0:  # Only calculate non-local energy if necessary
        Enonloc = calc_Enonloc(atoms, W)
    Ecoul = np.real(0.5 * n.conj().T @ atoms.Jdag(atoms.O(phi)))
    Exc = np.real(n.conj().T @ atoms.Jdag(atoms.O(atoms.J(exc))))
    if atoms.verbose >= 5 or out:
        print(f'Kinetic energy:              {Ekin:+.9f} Eh')
        print(f'Local potential energy:      {Eloc:+.9f} Eh')
        print(f'Non-local potential energy:  {Enonloc:+.9f} Eh')
        print(f'Coulomb energy:              {Ecoul:+.9f} Eh')
        print(f'Exchange-correlation energy: {Exc:+.9f} Eh')
    return Ekin + Eloc + Enonloc + Ecoul + Exc


# FIXME: Dimensions are scuffed
def getEwald(atoms):
    '''Calculate the Ewald/Coulomb energy.'''
    dr = norm(atoms.r - np.sum(atoms.R, axis=1) / 2, axis=1)
    sigma1 = 0.25
    gauss = 0
    # FIXME: wrong
    for Z in atoms.Z:
        g = np.exp(-dr**2 / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)**3
        gauss += Z * (np.sum(g) * det(atoms.R) / np.prod(atoms.S)) * g
    n = atoms.I(atoms.J(gauss) * atoms.Sf)
    n = np.real(n)
    phi = atoms.I(atoms.Linv(-4 * np.pi * atoms.O(atoms.J(n))))
    phi = np.real(phi)
    Unum = 0.5 * np.real(atoms.J(phi).conj().T @ atoms.O(atoms.J(n)))

    Uself = 0
    for Z in atoms.Z:
        Uself += Z**2 / (2 * np.sqrt(np.pi)) * (1 / sigma1)
    return (Unum[0][0] - Uself)


def getgrad(atoms, W):
    '''Calculate the energy gradient with respect to W.'''
    U = W.conj().T @ atoms.O(W)
    invU = inv(U)
    HW = H(atoms, W)
    F = np.diag(atoms.f)
    U12 = sqrtm(inv(U))
    Ht = U12 @ (W.conj().T @ HW) @ U12
    return (HW - (atoms.O(W) @ invU) @ (W.conj().T @ HW)) @ (U12 @ F @ U12) + \
           atoms.O(W) @ (U12 @ Q(Ht @ F - F @ Ht, U))


# def getgrad(atoms, W):
#     grad = np.zeros(W.size)
#     H_psi = H(atoms, W)
#     for ist in range(atoms.Ns):
#         grad[:, ist] = H_psi[:, ist]
#         for range(atoms.Ns):
#             grad[:, ist] -= np.dot(psi[:, jst], H_psi[:, ist]) * psi[:, jst]
#         grad[:,ist] *= atoms.f[ist]
#
#     if np.all(atoms.f == 2) or np.all(atoms.f == 1):
#         return grad
#
#     F = np.diag(atoms.f)
#     H = W.T * H_psi
#     HFH = H*F - F*H
#     Q = 0.5 * HFH
#     grad[:,:] += psi * H
#     return grad


def getPsi(atoms, W):
    '''Calculate eigensolutions and eigenvalues from the coefficent matrix W.'''
    W = orth(atoms, W)
    mu = W.conj().T @ H(atoms, W)
    epsilon, D = eig(mu)
    return W @ D, np.real(epsilon)


def getn(atoms, W):
    '''Generate the electronic density.'''
    W = W.T
    n = np.zeros((np.prod(atoms.S), 1))
    for i in range(W.shape[0]):
        psi = atoms.I(W[i])
        n += atoms.f[i] * np.real(psi.conj() * psi)
    return n.T[0]


def sd(atoms, W, Nit, etol):
    '''Steepest descent minimization algorithm.'''
    print('Start steepest descent minimization...')
    Elist = []
    alpha = 3e-5
    for i in range(Nit):
        W = W - alpha * getgrad(atoms, W)
        E = getE(atoms, W)
        Elist.append(E)
        if atoms.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E:+.7f}')
        if i > 0:
            if abs(Elist[-2] - Elist[-1]) < etol:
                print(f'Converged after {i+1} steps.')
                break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def lm(atoms, W, Nit, etol):
    '''Line minimization algorithm.'''
    print('Start Line minimization...')
    Elist = []
    alphat = 3e-5
    g = getgrad(atoms, W)
    d = -g
    gt = getgrad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = getE(atoms, W)
    Elist.append(E)
    if atoms.verbose >= 3:
        print(f'Nit: 1  \tE(W): {E:+.7f}')
    for i in range(1, Nit):
        g = getgrad(atoms, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -g
        gt = getgrad(atoms, W + alphat * d)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = getE(atoms, W)
        Elist.append(E)
        if atoms.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E:+.7f}\tlinmin-test: {linmin:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def pclm(atoms, W, Nit, etol):
    '''Preconditioned line minimization algorithm.'''
    print('Start preconditioned line minimization...')
    Elist = []
    alphat = 3e-5
    g = getgrad(atoms, W)
    d = -atoms.K(g)
    gt = getgrad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = getE(atoms, W)
    Elist.append(E)
    if atoms.verbose >= 3:
        print(f'Nit: 1  \tE(W): {E:+.7f}')
    for i in range(1, Nit):
        g = getgrad(atoms, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -atoms.K(g)
        gt = getgrad(atoms, W + alphat * d)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = getE(atoms, W)
        Elist.append(E)
        if atoms.verbose >= 3:
            print(f'Nit: {i+1}  \tE(W): {E:+.7f}\tlinmin-test: {linmin:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def pccg(atoms, W, Nit, etol, cgform=1):
    '''Preconditioned conjugate-gradient algorithm.'''
    print('Start preconditioned conjugate-gradient minimization...')
    Elist = []
    alphat = 3e-5
    g = getgrad(atoms, W)
    d = -atoms.K(g)
    gt = getgrad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    dold = d
    gold = g
    E = getE(atoms, W)
    Elist.append(E)
    if atoms.verbose >= 3:
        print(f'Nit: 0  \tE(W): {E:+.7f}')
    for i in range(1, Nit):
        g = getgrad(atoms, W)
        linmin = dotprod(g, dold) / np.sqrt(dotprod(g, g) * dotprod(dold, dold))
        cg = dotprod(g, atoms.K(gold)) / np.sqrt(dotprod(g, atoms.K(g)) * dotprod(gold, atoms.K(gold)))
        if cgform == 1:
            beta = dotprod(g, atoms.K(g)) / dotprod(gold, atoms.K(gold))
        elif cgform == 2:
            beta = dotprod(g - gold, atoms.K(g)) / dotprod(gold, atoms.K(gold))
        elif cgform == 3:
            beta = dotprod(g - gold, atoms.K(g)) / dotprod(g - gold, dold)
        d = -atoms.K(g) + beta * dold
        gt = getgrad(atoms, W + alphat * d)
        # FIXME: This feels wrong
        # If this becomes zero, the result will become nonsense
        if abs(dotprod(g - gt, d)) == 0:
            break
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        dold = d
        gold = g
        E = getE(atoms, W)
        Elist.append(E)
        if atoms.verbose >= 3:
            print(f'Nit: {i}  \tE(W): {E:+.7f}\tlinmin-test: {linmin:+.7f} \tcg-test: {cg:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def excVWN(n):
    '''VWN parameterization of the exchange correlation energy functional.'''
    X1 = 0.75 * (3 / (2 * np.pi))**(2 / 3)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c
    rs = (4 * np.pi / 3 * n)**(-1 / 3)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    out = -X1 / rs + A * (np.log(x * x / X) + 2 * b / Q * np.arctan(Q / (2 * x + b)) -
          (b * x0) / X0 * (np.log((x - x0) * (x - x0) / X) + 2 * (2 * x0 + b) / Q *
          np.arctan(Q / (2 * x + b))))
    return out


def excpVWN(n):
    '''Derivation with respect to n of the VWN exchange correlation energy functional.'''
    X1 = 0.75 * (3 / (2 * np.pi))**(2 / 3)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c
    rs = (4 * np.pi / 3 * n)**(-1 / 3)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    dx = 0.5 / x
    out = dx * (2 * X1 / (rs * x) + A * (2 / x - (2 * x + b) / X - 4 * b / (Q * Q + (2 * x + b) *
          (2 * x + b)) - (b * x0) / X0 * (2 / (x - x0) - (2 * x + b) / X - 4 * (2 * x0 + b) /
          (Q * Q + (2 * x + b) * (2 * x + b)))))
    return (-rs / (3 * n)) * out


def orth(atoms, W):
    '''Orthogonalize coefficent matrix W.'''
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


# FIXME: Only for testing
# def orth2(atoms, W):
#     '''Orthogonalize coefficent matrix W.'''
#     return W @ inv(sqrtm(W.conj().T @ W))


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.'''
    mu, V = eig(U)
    mu = np.reshape(mu, (len(mu), 1))
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T
