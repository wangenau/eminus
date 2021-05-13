#!/usr/bin/env python3
'''
Main SCF file with every relevant function.
'''
import numpy as np
from numpy.linalg import eig, inv#, det
from numpy.random import randn
from scipy.linalg import sqrtm
from timeit import default_timer
from .energies import get_Ekin, get_Eloc, get_Enonloc, get_Ecoul, get_Exc, get_Eewald
from .lda_VWN import excVWN, excpVWN#, xc_vwn
from .utils import Diagprod, dotprod
from .gth_nonloc import calc_Vnonloc


def SCF(atoms, n_sd=10, n_lm=0, n_pclm=0, n_cg=100, cgform=1, etol=1e-7):
    '''Main SCF function.'''
    # Set up basis functions
    # Start with randomized, complex, orthogonal basis functions
    # from numpy.random import seed
    # seed(1234)  # Uncomment for a fixed starting point
    W = randn(len(atoms.active[0]), atoms.Ns) + 1j * randn(len(atoms.active[0]), atoms.Ns)

    # Minimization procedure
    start = default_timer()
    if n_sd > 0:
        W = orth(atoms, W)
        W, Elist = sd(atoms, W, n_sd, etol)
    if n_lm > 0:
        W = orth(atoms, W)
        W, Elist = lm(atoms, W, n_lm, etol)
    if n_pclm > 0:
        W = orth(atoms, W)
        W, Elist = pclm(atoms, W, n_pclm, etol)
    if n_cg > 0:
        W = orth(atoms, W)
        W, Elist = pccg(atoms, W, n_cg, etol, cgform)
    end = default_timer()
    # Handle energies and output
    Eel = Elist[-1]
    Eewald = get_Eewald(atoms)
    Etot = Eel + Eewald
    if atoms.verbose >= 5:
        print(f'Compression: {len(atoms.G2) / len(atoms.G2c):.5f}')
    if atoms.verbose >= 4:
        print(f'Time spent: {end - start:.5f}s')
    if atoms.verbose >= 3:
        get_E(atoms, W, True)
        print(f'Electronic energy:           {Eel:+.9f} Eh')
        print(f'Ewald energy:                {Eewald:+.9f} Eh')
    print(f'Total energy:                {Etot:+.9f} Eh')

    # Save calculation parameters
    W = orth(atoms, W)
    atoms.W = W
    atoms.psi, atoms.estate = get_psi(atoms, W)  # Save wave functions and
    atoms.n = get_n_total(atoms, W)  # Save electronic density
    atoms.etot = Etot  # Save total energy
    return


def H(atoms, W):
    '''Left-hand side of our eigenvalue equation.'''
    Y = orth(atoms, W)  # Orthogonalize at the start
    n = get_n_total(atoms, Y)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    exc = excVWN(n)
    excp = excpVWN(n)
    Vdual = atoms.Vloc
    Veff = Vdual + atoms.Jdag(atoms.O(phi)) + atoms.Jdag(atoms.O(atoms.J(exc))) + \
           excp * atoms.Jdag(atoms.O(atoms.J(n)))
    #Veff = Vdual + atoms.Jdag(atoms.O(phi))
    #Vxc_psi = xc_vwn(n)[1]
    Vnonloc_psi = 0
    if atoms.NbetaNL > 0:  # Only calculate non-local potential if necessary
        Vnonloc_psi = calc_Vnonloc(atoms, W)
    Vkin_psi = -0.5 * atoms.L(W)# / det(atoms.R)  #TODO: Divide by det? also presort of G2c is necessary
    #Vkin_psi = -0.5 * atoms.L(W) / det(atoms.R)
    return Vkin_psi + atoms.Idag(Diagprod(Veff, atoms.I(W))) + Vnonloc_psi# + Vxc_psi

# a=randn(prod(S),1)+i*randn(prod(S),1)
# b=randn(prod(S),1)+i*randn(prod(S),1)
# conj(a’*H(b))
# b’*H(a)


def get_E(atoms, W, out=False):
    '''Calculate the sum of energies over Ns states.'''
    Y = orth(atoms, W)
    n = get_n_total(atoms, Y)
    Ekin = get_Ekin(atoms, Y)
    Eloc = get_Eloc(atoms, n)
    Enonloc = get_Enonloc(atoms, W)
    Ecoul = get_Ecoul(atoms, n)
    Exc = get_Exc(atoms, n)
    if atoms.verbose >= 5 or out:
        print(f'Kinetic energy:              {Ekin:+.9f} Eh')
        print(f'Local potential energy:      {Eloc:+.9f} Eh')
        print(f'Non-local potential energy:  {Enonloc:+.9f} Eh')
        print(f'Coulomb energy:              {Ecoul:+.9f} Eh')
        print(f'Exchange-correlation energy: {Exc:+.9f} Eh')
    return Ekin + Eloc + Enonloc + Ecoul + Exc


def get_grad(atoms, W):
    '''Calculate the energy gradient with respect to W.'''
    U = W.conj().T @ atoms.O(W)
    invU = inv(U)
    HW = H(atoms, W)
    F = np.diag(atoms.f)
    U12 = sqrtm(inv(U))
    Ht = U12 @ (W.conj().T @ HW) @ U12
    return (HW - (atoms.O(W) @ invU) @ (W.conj().T @ HW)) @ (U12 @ F @ U12) + \
           atoms.O(W) @ (U12 @ Q(Ht @ F - F @ Ht, U))


def get_psi(atoms, W):
    '''Calculate eigensolutions and eigenvalues from the coefficent matrix W.'''
    Y = orth(atoms, W)
    mu = Y.conj().T @ H(atoms, Y)
    epsilon, D = eig(mu)
    return Y @ D, np.real(epsilon)


def get_n_total(atoms, W):
    '''Generate the total electronic density.'''
    W = W.T
    n = np.zeros((np.prod(atoms.S), 1))
    for i in range(W.shape[0]):
        psi = atoms.I(W[i])
        n += atoms.f[i] * np.real(psi.conj() * psi)
    return n.T[0]


def get_n_single(atoms, W):
    '''Generate single electronic densities.'''
    W = W.T
    n = np.zeros((np.prod(atoms.S), len(W)))
    for i in range(W.shape[0]):
        psi = atoms.I(W[i])
        n[:, i] = atoms.f[i] * np.real(psi.conj() * psi).T
    return n.T


def sd(atoms, W, Nit, etol):
    '''Steepest descent minimization algorithm.'''
    print('Start steepest descent minimization...')
    Elist = []
    alpha = 3e-5
    for i in range(Nit):
        W = W - alpha * get_grad(atoms, W)
        E = get_E(atoms, W)
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
    g = get_grad(atoms, W)
    d = -g
    gt = get_grad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = get_E(atoms, W)
    Elist.append(E)
    if atoms.verbose >= 3:
        print(f'Nit: 1  \tE(W): {E:+.7f}')
    for i in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -g
        gt = get_grad(atoms, W + alphat * d)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = get_E(atoms, W)
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
    g = get_grad(atoms, W)
    d = -atoms.K(g)
    gt = get_grad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = get_E(atoms, W)
    Elist.append(E)
    if atoms.verbose >= 3:
        print(f'Nit: 1  \tE(W): {E:+.7f}')
    for i in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -atoms.K(g)
        gt = get_grad(atoms, W + alphat * d)
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        E = get_E(atoms, W)
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
    g = get_grad(atoms, W)
    d = -atoms.K(g)
    gt = get_grad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    dold = d
    gold = g
    E = get_E(atoms, W)
    Elist.append(E)
    if atoms.verbose >= 3:
        print(f'Nit: 0  \tE(W): {E:+.7f}')
    for i in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, dold) / np.sqrt(dotprod(g, g) * dotprod(dold, dold))
        cg = dotprod(g, atoms.K(gold)) / np.sqrt(dotprod(g, atoms.K(g)) * \
             dotprod(gold, atoms.K(gold)))
        if cgform == 1:
            beta = dotprod(g, atoms.K(g)) / dotprod(gold, atoms.K(gold))
        elif cgform == 2:
            beta = dotprod(g - gold, atoms.K(g)) / dotprod(gold, atoms.K(gold))
        elif cgform == 3:
            beta = dotprod(g - gold, atoms.K(g)) / dotprod(g - gold, dold)
        d = -atoms.K(g) + beta * dold
        gt = get_grad(atoms, W + alphat * d)
        # FIXME: This feels wrong
        # If this becomes zero, the result will become nonsense
        if abs(dotprod(g - gt, d)) == 0:
            break
        alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
        W = W + alpha * d
        dold = d
        gold = g
        E = get_E(atoms, W)
        Elist.append(E)
        if atoms.verbose >= 3:
            print(f'Nit: {i}  \tE(W): {E:+.7f}\tlinmin-test: {linmin:+.7f} \tcg-test: {cg:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def orth(atoms, W):
    '''Orthogonalize coefficent matrix W.'''
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.'''
    mu, V = eig(U)
    mu = np.reshape(mu, (len(mu), 1))
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T
