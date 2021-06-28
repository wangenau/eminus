#!/usr/bin/env python3
'''
SCF function with every relevant function.
'''
from timeit import default_timer

import numpy as np
from numpy.linalg import eig, inv, norm
from numpy.random import randn, seed
from scipy.linalg import sqrtm

from .energies import get_Ecoul, get_Eewald, get_Ekin, get_Eloc, get_Enonloc, get_Exc
from .exc import exc_vwn, excp_vwn
from .gth import calc_Vnonloc
from .utils import Diagprod, dotprod


def SCF(atoms, guess='random', etol=1e-7, n_sd=10, n_lm=0, n_pclm=0, n_cg=100, cgform=1):
    '''Main SCF function to do direct minimizations.

    Args:
        atoms :

    Kwargs:
        guess :

        etol :

        cgform :

    Returns:
        Total energy as a float.
    '''
    # Update atoms object at the beginning to ensure correct inputs
    atoms.update()

    # Set up basis functions
    guess = guess.lower()
    if guess == 'gauss' or guess == 'gaussian':
        # Start with gaussians at atom positions
        W = guess_gaussian(atoms)
    else:
        # Start with randomized, complex basis functions with a random seed
        W = guess_random(atoms, complex=True, reproduce=False)

    # Calculate ewald energy
    atoms.energies.Eewald = get_Eewald(atoms)

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

    # Handle output
    if atoms.verbose >= 5:
        print(f'Compression: {len(atoms.G2) / len(atoms.G2c):.5f}')
    if atoms.verbose >= 4:
        print(f'Time spent: {end - start:.5f}s')
    if atoms.verbose >= 3:
        print('Energy contributions:')
        print(atoms.energies)
    else:
        print(f'Total energy: {atoms.energies.Etot:.9f} Eh')

    # Save basis functions
    atoms.W = orth(atoms, W)
    return atoms.energies.Etot


def H(atoms, W):
    '''Left-hand side of the eigenvalue equation.'''
    Y = orth(atoms, W)  # Orthogonalize at the start
    n = get_n_total(atoms, Y)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    exc = exc_vwn(n)
    excp = excp_vwn(n)

    # Calculate the effective potential, with or without Coulomb truncation
    Veff = atoms.Vloc + atoms.Jdag(atoms.O(atoms.J(exc))) + excp * atoms.Jdag(atoms.O(atoms.J(n)))
    if atoms.cutcoul is None:
        Veff += atoms.Jdag(atoms.O(phi))
    else:
        Rc = atoms.cutcoul
        correction = np.cos(np.sqrt(atoms.G2) * Rc) * atoms.O(phi)
        Veff += atoms.Jdag(atoms.O(phi) - correction)

    Vkin_psi = -0.5 * atoms.L(W)
    Vnonloc_psi = calc_Vnonloc(atoms, W)
    return Vkin_psi + atoms.Idag(Diagprod(Veff, atoms.I(W))) + Vnonloc_psi


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.'''
    mu, V = eig(U)
    mu = np.reshape(mu, (len(mu), 1))
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T


def get_E(atoms, W):
    '''Calculate all the energy contributions.'''
    Y = orth(atoms, W)
    n = get_n_total(atoms, Y)
    atoms.energies.Ekin = get_Ekin(atoms, Y)
    atoms.energies.Eloc = get_Eloc(atoms, n)
    atoms.energies.Enonloc = get_Enonloc(atoms, Y)
    atoms.energies.Ecoul = get_Ecoul(atoms, n)
    atoms.energies.Exc = get_Exc(atoms, n)
    if atoms.verbose >= 5:
        print(atoms.energies)
    return atoms.energies.Etot


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
            print(f'Nit: {i+1}  \tEtot: {atoms.energies.Etot:+.7f}')
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
        print(f'Nit: 1  \tEtot: {atoms.energies.Etot:+.7f}')
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
            print(f'Nit: {i+1}  \tEtot: {atoms.energies.Etot:+.7f}\tlinmin-test: {linmin:+.7f}')
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
        print(f'Nit: 1  \tEtot: {atoms.energies.Etot:+.7f}')
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
            print(f'Nit: {i+1}  \tEtot: {atoms.energies.Etot:+.7f}\tlinmin-test: {linmin:+.7f}')
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
        print(f'Nit: 1  \tEtot: {atoms.energies.Etot:+.7f}')
    for i in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, dold) / np.sqrt(dotprod(g, g) * dotprod(dold, dold))
        cg = dotprod(g, atoms.K(gold)) / np.sqrt(dotprod(g, atoms.K(g)) *
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
            print(f'Nit: {i+1}  \tEtot: {atoms.energies.Etot:+.7f}'
                  f'\tlinmin-test: {linmin:+.7f} \tcg-test: {cg:+.7f}')
        if abs(Elist[-2] - Elist[-1]) < etol:
            print(f'Converged after {i+1} steps.')
            break
    if abs(Elist[-2] - Elist[-1]) > etol:
        print('Not converged!')
    return W, np.asarray(Elist)


def orth(atoms, W):
    '''Orthogonalize coefficent matrix W.'''
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def get_psi(atoms, Y):
    '''Calculate eigensolutions and eigenvalues from the coefficent matrix W.'''
    mu = Y.conj().T @ H(atoms, Y)
    epsilon, D = eig(mu)
    return Y @ D, np.real(epsilon)


def get_n_total(atoms, Y):
    '''Calculate the total electronic density.'''
    Y = Y.T
    n = np.zeros((np.prod(atoms.S), 1))
    for i in range(Y.shape[0]):
        psi = atoms.I(Y[i])
        n += atoms.f[i] * np.real(psi.conj() * psi)
    return n.T[0]


def get_n_single(atoms, Y):
    '''Calculate the single electronic densities.'''
    Y = Y.T
    n = np.zeros((np.prod(atoms.S), len(Y)))
    for i in range(Y.shape[0]):
        psi = atoms.I(Y[i])
        n[:, i] = atoms.f[i] * np.real(psi.conj() * psi).T
    return n.T


def guess_random(atoms, complex=True, reproduce=False):
    '''Generate random coefficents as starting values.'''
    if reproduce:
        seed(42)
    if complex:
        return randn(len(atoms.active[0]), atoms.Ns) + 1j * randn(len(atoms.active[0]), atoms.Ns)
    else:
        return randn(len(atoms.active[0]), atoms.Ns)


def guess_gaussian(atoms):
    '''Generate inital-guess coefficents using normalized Gaussians as starting values.'''
    sigma = 0.5
    normal = (2 * np.pi * sigma**2)**(3 / 2)

    W = np.zeros((len(atoms.r), atoms.Ns))
    for ist in range(atoms.Ns):
        # If we have more states than atoms, start all over again
        ia = ist % len(atoms.X)
        r = norm(atoms.r - atoms.X[ia], axis=1)
        W[:, ist] = atoms.Z[ia] * np.exp(-r**2 / (2 * sigma**2)) / normal
    # Transform from real-space to reciprocal space
    # There is no transformation on the active space for this, so do it "manually"
    return atoms.J(W)[atoms.active]
