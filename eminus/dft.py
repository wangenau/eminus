#!/usr/bin/env python3
'''Main DFT functions based on the DFT++ formulation.

Reference: Comput. Phys. Commun. 128, 1.
'''
import numpy as np
from numpy.linalg import eig, eigh, inv, norm
from numpy.random import randn, seed
from scipy.linalg import sqrtm

from .gth import calc_Vnonloc
from .utils import diagprod
from .xc import get_xc


def solve_poisson(atoms, n):
    '''Solve the Poisson equation.

    Args:
        atoms: Atoms object.
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Hartree field.
    '''
    # phi = -4 pi Linv(O(J(n)))
    return -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))


def get_n_total(atoms, Y):
    '''Calculate the total electronic density.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        ndarray: Electronic density.
    '''
    # n = (IW) F (IW)dag
    Yrs = atoms.I(Y)
    n = atoms.f * np.real(Yrs.conj() * Yrs)
    return np.sum(n, axis=1)


def get_n_single(atoms, Y):
    '''Calculate the single-electron densities.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        ndarray: Single-electron densities.
    '''
    Yrs = atoms.I(Y)
    return atoms.f * np.real(Yrs.conj() * Yrs)


def orth(atoms, W):
    '''Orthogonalize coefficient matrix W.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Orthogonalized wave functions.
    '''
    # Y = W (Wdag O(W))^-0.5
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def get_grad(scf, W, Y=None, n=None, phi=None, vxc=None):
    '''Calculate the energy gradient with respect to W.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.
        n (ndarray): Real-space electronic density.
        phi (ndarray): Hartree field.
        vxc (ndarray): Exchange-correlation potential.

    Returns:
        ndarray: Gradient.
    '''
    atoms = scf.atoms
    F = np.diag(atoms.f)
    HW = H(scf, W, Y, n, phi, vxc)
    WHW = W.conj().T @ HW
    # U = Wdag O(W)
    U = W.conj().T @ atoms.O(W)
    invU = inv(U)
    U12 = sqrtm(invU)
    # Htilde = U^-0.5 Wdag H(W) U^-0.5
    Ht = U12 @ WHW @ U12
    # grad E = H(W) - O(W)U^-1 (Wdag H(W))(U^-0.5 F U^-0.5) + O(W) (U^-0.5 Q(Htilde F - F Htilde))
    return (HW - (atoms.O(W) @ invU) @ WHW) @ (U12 @ F @ U12) + \
           atoms.O(W) @ (U12 @ Q(Ht @ F - F @ Ht, U))


def H(scf, W, Y=None, n=None, phi=None, vxc=None):
    '''Left-hand side of the eigenvalue equation.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.
        n (ndarray): Real-space electronic density.
        phi (ndarray): Hartree field.
        vxc (ndarray): Exchange-correlation potential.

    Returns:
        ndarray: Hamiltonian applied on W.
    '''
    atoms = scf.atoms

    # One can calculate everything from W,
    # but one can also use already computed results to save time
    if Y is None:
        Y = orth(atoms, W)
    if n is None:
        n = get_n_total(atoms, Y)
    if phi is None:
        phi = solve_poisson(atoms, n)
    if vxc is None:
        vxc = get_xc(scf.xc, n)[1]

    # We get the full potential in the functional definition (different to the DFT++ notation)
    # Normally Vxc = Jdag(O(J(exc))) + diag(exc')Jdag(O(J(n)))
    Vxc = atoms.Jdag(atoms.O(atoms.J(vxc)))
    # Vkin = -0.5 L(W)
    Vkin_psi = -0.5 * atoms.L(W)
    # Veff = Jdag(Vion) + Jdag(O(J(vxc))) + Jdag(O(phi))
    Veff = scf.Vloc + Vxc + atoms.Jdag(atoms.O(phi))
    Vnonloc_psi = calc_Vnonloc(scf, W)
    # H = Vkin + Idag(diag(Veff))I + Vnonloc
    return Vkin_psi + atoms.Idag(diagprod(Veff, atoms.I(W))) + Vnonloc_psi


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.

    Args:
        inp (ndarray): Coefficients input array.
        U (ndarray): Overlap of wave functions.

    Returns:
        ndarray: Q operator result.
    '''
    mu, V = eig(U)
    mu = mu[:, None]
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom) @ V.conj().T


def get_psi(scf, Y, n=None):
    '''Calculate eigenstates from H.

    Args:
        scf: SCF object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Eigenstates in reciprocal space.
    '''
    mu = Y.conj().T @ H(scf, W=Y, n=n)
    _, D = eigh(mu)
    return Y @ D


def get_epsilon(scf, Y, n=None):
    '''Calculate eigenvalues from H.

    Args:
        scf: SCF object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Eigenvalues.
    '''
    mu = Y.conj().T @ H(scf, W=Y, n=n)
    epsilon, _ = eigh(mu)
    return np.sort(epsilon)


def guess_random(scf, complex=True, reproduce=True):
    '''Generate random initial-guess coefficients as starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        complex (bool): Use complex numbers for the random guess.
        reproduce (bool): Use a set seed for reproducible random numbers.

    Returns:
        ndarray: Initial-guess orthogonal wave functions in reciprocal space.
    '''
    atoms = scf.atoms
    if reproduce:
        seed(42)
    if complex:
        W = randn(len(atoms.G2c), atoms.Ns) + 1j * randn(len(atoms.G2c), atoms.Ns)
    else:
        W = randn(len(atoms.G2c), atoms.Ns)
    return orth(atoms, W)


def guess_gaussian(scf):
    '''Generate initial-guess coefficients using normalized Gaussians as starting values.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Initial-guess orthogonal wave functions in reciprocal space.
    '''
    atoms = scf.atoms
    # Start with randomized wave functions
    W = guess_random(scf, complex=True, reproduce=True)

    sigma = 0.5
    normal = (2 * np.pi * sigma**2)**(3 / 2)
    # Calculate a density from normalized Gauss functions
    n = np.zeros(len(atoms.r))
    for ia in range(atoms.Natoms):
        r = norm(atoms.r - atoms.X[ia], axis=1)
        n += atoms.Z[ia] * np.exp(-r**2 / (2 * sigma**2)) / normal
    # Calculate the eigenfunctions
    return get_psi(scf, W, n)
