#!/usr/bin/env python3
'''
SCF calculation functionalities with relevant sub-function.
'''
from timeit import default_timer

import numpy as np
from numpy.linalg import eig, inv, norm
from numpy.random import randn, seed
from scipy.linalg import sqrtm

from .energies import get_Ecoul, get_Eewald, get_Ekin, get_Eloc, get_Enonloc, get_Exc
from .exc import get_exc
from .gth import calc_Vnonloc
from .utils import Diagprod, dotprod


def SCF(atoms, guess='gaussian', etol=1e-7, min=None, cgform=1):
    '''SCF function to handle direct minimizations.

    Args:
        atoms :
            Atoms object.

    Kwargs:
        guess : str
            Initial guess method for the basis functions (case insensitive).
            Example: 'Gauss'; 'gaussian'; 'random'; 'rand'; 'Experimental'; 'exp'
            Default: 'gaussian'

        etol : float
            Convergence tolerance of the total energy.
            Default: 1e-7

        min : dict
            Dictionary to set the maximum amount of steps per minimization method and their order.
            Example: {'sd': 10, 'pccg': 100}; {'pccg': 10, 'lm': 25, 'pclm': 50}
            Default: None (will default to {'pccg': 100})

        cgform : int
            Conjugated-gradient from for the preconditioned conjugate-gradient minimization (pccg).
            1 for Fletcher-Reeves, 2 for Polak-Ribiere, and 3 for Hestenes-Stiefel.
            Default: 1

    Returns:
        Total energy as a float.
    '''
    # Map minimization names and functions, also use this dict to save times and iterations
    minimizer = {
        'sd': {
            'func': sd,
            'name': 'steepest descent minimization'
        },
        'lm': {
            'func': lm,
            'name': 'line minimization'
        },
        'pclm': {
            'func': pclm,
            'name': 'preconditioned line minimization'
        },
        'pccg': {
            'func': pccg,
            'name': 'preconditioned conjugate-gradient minimization'
        }
    }

    # Set min here, better not use mutable data types in signatures
    if min is None:
        min = {'pccg': 100}

    # Print some useful informations
    if atoms.verbose >= 3:
        print(f'--- System informations ---\n{atoms}\n')
    if atoms.verbose >= 4:
        print(f'--- Cell informations ---\nSide lengths: {atoms.a} Bohr\n'
              f'Cut-off energy: {atoms.ecut} Hartree\n'
              f'Sampling per axis: ({atoms.S[0]}, {atoms.S[1]}, {atoms.S[2]})\n')
        print(f'--- Calculation informations ---\nSpin polarization: {atoms.spinpol}\n'
              f'Number of states: {atoms.Ns}\n'
              f'Occupation per state: {atoms.f}\n'
              f'Potential: {atoms.pot}\n'
              f'Non-local contribution: {atoms.NbetaNL > 0}\n'
              f'Coulomb-truncation: {atoms.cutcoul is not None}\n'
              f'XC functionals: {atoms.exc}\n'
              f'Compression: {len(atoms.G2) / len(atoms.G2c):.5f}\n')

    # Set up basis functions
    guess = guess.lower()
    if guess in ('gauss', 'gaussian'):
        # Start with gaussians at atom positions
        W = guess_gaussian(atoms)
    elif guess in ('rand', 'random'):
        # Start with randomized, complex basis functions with a random seed
        W = guess_random(atoms, complex=True, reproduce=True)
    elif guess in ('exp', 'experimental'):
        # Start with s-type orbitals. This may work good for small systems.
        W = guess_experimental(atoms)

    # Calculate ewald energy
    atoms.energies.Eewald = get_Eewald(atoms)

    # Start minimization procedures
    if atoms.verbose >= 3:
        print('--- SCF data ---')
    Etots = []
    for imin in min:
        start = default_timer()
        if atoms.verbose >= 2:
            print(f'Start {minimizer[imin]["name"]}...')
        W, Elist = minimizer[imin]['func'](atoms, W, min[imin], etol, cgform=cgform)
        end = default_timer()
        minimizer[imin]['time'] = end - start
        minimizer[imin]['iteration'] = len(Elist)
        Etots += Elist
    if atoms.verbose >= 1:
        if abs(Etots[-2] - Etots[-1]) < etol:
            print(f'SCF converged after {len(Etots)} iterations.')
        else:
            print('SCF not converged!')

    # Print SCF data
    if atoms.verbose >= 2:
        if atoms.verbose >= 3:
            print(f'\n--- SCF results ---\nStarting guess: {guess}')
        t_total = 0
        for imin in min:
            N = minimizer[imin]['iteration']
            t = minimizer[imin]['time']
            t_total += t
            if atoms.verbose >= 4:
                print(f'\nMinimizer: {imin}'
                      f'\nIterations:\t{N}'
                      f'\nTime:\t\t{t:.5f}s'
                      f'\nTime/Iteration:\t{t / N:.5f}s')
        print(f'Total SCF time: {t_total:.5f}s')

    # Print energy data
    if atoms.verbose >= 3:
        print('\n--- Energy data ---')
    if atoms.verbose >= 4:
        print(f'{atoms.energies}')
    elif atoms.verbose >= 1:
        print(f'Total energy: {atoms.energies.Etot:.9f} Eh')

    # Save basis functions and density to reuse them later
    atoms.W = orth(atoms, W)
    atoms.n = get_n_total(atoms, atoms.W)
    return atoms.energies.Etot


def H(atoms, W, n=None):
    '''Left-hand side of the eigenvalue equation.'''
    Y = orth(atoms, W)  # Orthogonalize at the start
    if n is None:
        n = get_n_total(atoms, Y)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))

    # Calculate the exchange-correlation potential
    # We get the full potential in the functinal definition, unlike in the Arias notation
    vxc = get_exc(atoms.exc, n, atoms.spinpol)[1]
    Vxc = atoms.Jdag(atoms.O(atoms.J(vxc)))

    # Calculate the effective potential, with or without Coulomb truncation
    if atoms.cutcoul is None:
        Veff = atoms.Vloc + Vxc + atoms.Jdag(atoms.O(phi))
    else:
        Rc = atoms.cutcoul
        correction = np.cos(np.sqrt(atoms.G2) * Rc) * atoms.O(phi)
        Veff = atoms.Vloc + Vxc + atoms.Jdag(atoms.O(phi) - correction)

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


def check_energies(atoms, Elist, etol, linmin=None, cg=None):
    '''Check the energies for every SCF cycle and handle the output.'''
    Nit = len(Elist)

    # Output handling
    if linmin is not None:
        linmin = f'\tlinmin-test: {linmin:+.7f}'
    else:
        linmin = ''
    if cg is not None:
        cg = f'\tcg-test: {cg:+.7f}'
    else:
        cg = ''

    if atoms.verbose >= 5:
        print(f'Iteration: {Nit} {linmin} {cg}\n{atoms.energies}\n')
    elif atoms.verbose >= 4:
        print(f'Iteration: {Nit}  \tEtot: {atoms.energies.Etot:+.7f} {linmin} {cg}')
    elif atoms.verbose >= 3:
        print(f'Iteration: {Nit}  \tEtot: {atoms.energies.Etot:+.7f}')

    # Check for convergence
    if Nit > 1:
        if abs(Elist[-2] - Elist[-1]) < etol:
            return True
        if Elist[-1] > Elist[-2] and atoms.verbose >= 3:
            print('WARNING: Total energy is not decreasing.')
    return False


def sd(atoms, W, Nit, etol, **kwargs):
    '''Steepest descent minimization algorithm.'''
    Elist = []
    alpha = 3e-5

    for _ in range(Nit):
        W = W - alpha * get_grad(atoms, W)
        E = get_E(atoms, W)
        Elist.append(E)
        if check_energies(atoms, Elist, etol):
            break
    return W, Elist


def lm(atoms, W, Nit, etol, **kwargs):
    '''Line minimization algorithm.'''
    Elist = []
    alphat = 3e-5

    # Do the first step without the linmin test
    g = get_grad(atoms, W)
    d = -g
    gt = get_grad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = get_E(atoms, W)
    Elist.append(E)
    check_energies(atoms, Elist, etol)

    for _ in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -g
        gt = get_grad(atoms, W + alphat * d)
        # The denominator can get zero, add this check to prevent this
        denom = dotprod(g - gt, d)
        if abs(denom) > 1e-16:  # 1e-16 is the range under float64 machine precision
            alpha = alphat * dotprod(g, d) / denom
        else:
            alpha = alphat * dotprod(g, d) / 1e-16
        W = W + alpha * d
        E = get_E(atoms, W)
        Elist.append(E)
        if check_energies(atoms, Elist, etol, linmin):
            break
    return W, Elist


def pclm(atoms, W, Nit, etol, **kwargs):
    '''Preconditioned line minimization algorithm.'''
    Elist = []
    alphat = 3e-5

    # Do the first step without the linmin test
    g = get_grad(atoms, W)
    d = -atoms.K(g)
    gt = get_grad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    E = get_E(atoms, W)
    Elist.append(E)
    check_energies(atoms, Elist, etol)

    for _ in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -atoms.K(g)
        gt = get_grad(atoms, W + alphat * d)
        # The denominator can get zero, add this check to prevent this
        denom = dotprod(g - gt, d)
        if abs(denom) > 1e-16:  # 1e-16 is the range under float64 machine precision
            alpha = alphat * dotprod(g, d) / denom
        else:
            alpha = alphat * dotprod(g, d) / 1e-16
        W = W + alpha * d
        E = get_E(atoms, W)
        Elist.append(E)
        if check_energies(atoms, Elist, etol, linmin):
            break
    return W, Elist


def pccg(atoms, W, Nit, etol, cgform=1):
    '''Preconditioned conjugate-gradient algorithm.'''
    Elist = []
    alphat = 3e-5

    # Do the first step without the linmin and cg test
    g = get_grad(atoms, W)
    d = -atoms.K(g)
    gt = get_grad(atoms, W + alphat * d)
    alpha = alphat * dotprod(g, d) / dotprod(g - gt, d)
    W = W + alpha * d
    dold = d
    gold = g
    E = get_E(atoms, W)
    Elist.append(E)
    check_energies(atoms, Elist, etol)

    for _ in range(1, Nit):
        g = get_grad(atoms, W)
        linmin = dotprod(g, dold) / np.sqrt(dotprod(g, g) * dotprod(dold, dold))
        cg = dotprod(g, atoms.K(gold)) / np.sqrt(dotprod(g, atoms.K(g)) *
             dotprod(gold, atoms.K(gold)))
        if cgform == 1:  # Fletcher-Reeves
            beta = dotprod(g, atoms.K(g)) / dotprod(gold, atoms.K(gold))
        elif cgform == 2:  # Polak-Ribiere
            beta = dotprod(g - gold, atoms.K(g)) / dotprod(gold, atoms.K(gold))
        elif cgform == 3:  # Hestenes-Stiefel
            beta = dotprod(g - gold, atoms.K(g)) / dotprod(g - gold, dold)
        d = -atoms.K(g) + beta * dold
        gt = get_grad(atoms, W + alphat * d)
        # The denominator can get zero, add this check to prevent this
        denom = dotprod(g - gt, d)
        if abs(denom) > 1e-16:  # 1e-16 is the range under float64 machine precision
            alpha = alphat * dotprod(g, d) / denom
        else:
            alpha = alphat * dotprod(g, d) / 1e-16
        W = W + alpha * d
        dold = d
        gold = g
        E = get_E(atoms, W)
        Elist.append(E)
        if check_energies(atoms, Elist, etol, linmin, cg):
            break
    return W, Elist


def orth(atoms, W):
    '''Orthogonalize coefficent matrix W.'''
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def get_psi(atoms, Y, n=None):
    '''Calculate eigenstates from H.'''
    mu = Y.conj().T @ H(atoms, Y, n)
    _, D = eig(mu)
    return Y @ D


def get_epsilon(atoms, Y, n=None):
    '''Calculate eigenvalues from H.'''
    mu = Y.conj().T @ H(atoms, Y, n)
    epsilon, _ = eig(mu)
    epsilon = np.real(epsilon)
    return np.sort(epsilon)


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
    n = np.empty((np.prod(atoms.S), len(Y)))
    for i in range(Y.shape[0]):
        psi = atoms.I(Y[i])
        n[:, i] = atoms.f[i] * np.real(psi.conj() * psi).T
    return n.T


def guess_random(atoms, complex=True, reproduce=False):
    '''Generate random coefficents as starting values.'''
    if reproduce:
        seed(42)
    if complex:
        W = randn(len(atoms.active[0]), atoms.Ns) + 1j * randn(len(atoms.active[0]), atoms.Ns)
    else:
        W = randn(len(atoms.active[0]), atoms.Ns)
    return orth(atoms, W)


def guess_gaussian(atoms):
    '''Generate inital-guess coefficents using normalized Gaussians as starting values.'''
    # Start with a randomized basis set
    W = guess_random(atoms, complex=True, reproduce=True)
    W = orth(atoms, W)

    eps = 1e-9
    sigma = 0.5
    normal = (2 * np.pi * sigma**2)**(3 / 2)

    # Calculate a density from normalized Gauss functions
    n = np.zeros(len(atoms.r)) + eps  # We divide by n in exc functionals, so 0 would be bad
    for ia in range(atoms.Natoms):
        r = norm(atoms.r - atoms.X[ia], axis=1)
        n += atoms.Z[ia] * np.exp(-r**2 / (2 * sigma**2)) / normal

    # Calculate the eigenfunctions
    W = get_psi(atoms, W, n)
    return W


def guess_experimental(atoms):
    '''Generate a s-type basis set.'''
    eps = 1e-9
    sigma = 0.5
    normal = (2 * np.pi * sigma**2)**(3 / 2)

    W = np.empty((len(atoms.r), atoms.Ns))
    for ist in range(atoms.Ns):
        # If we have more states than atoms, start all over again
        ia = ist % atoms.Natoms
        r = norm(atoms.r - atoms.X[ia], axis=1)
        W[:, ist] = atoms.Z[ia] * np.exp(-r**2 / (2 * sigma**2)) / normal + eps
        # Vary each W minimally by eps with changing signs, such that Wdag*O(W) is not singulary
        eps *= -1
    # Transform from real-space to reciprocal space
    # There is no transformation on the active space for this, so do it "manually"
    W = atoms.J(W)[atoms.active]
    return orth(atoms, W)
