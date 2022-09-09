#!/usr/bin/env python3
'''Main DFT functions based on the DFT++ formulation.

Reference: Comput. Phys. Commun. 128, 1.
'''
import numpy as np
from numpy.random import randn, seed
from scipy.linalg import eig, eigh, eigvalsh, inv, norm, sqrtm

from .gth import calc_Vnonloc
from .utils import diagprod, handle_spin_gracefully
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
    n = np.zeros(len(atoms.r))
    for spin in range(atoms.Nspin):
        n += np.sum(atoms.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin]), axis=1)
    return n


def get_n_spin(atoms, Y, n=None):
    '''Calculate the electronic density per spin channel.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Electronic density per spin.
    '''
    # Return the total density in the spin-paired case
    if n is not None and atoms.Nspin == 1:
        return np.atleast_2d(n)

    Yrs = atoms.I(Y)
    n = np.empty((atoms.Nspin, len(atoms.r)))
    for spin in range(atoms.Nspin):
        n[spin] = np.sum(atoms.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin]), axis=1)
    return n


def get_n_single(atoms, Y):
    '''Calculate the single-electron densities.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        ndarray: Single-electron densities.
    '''
    Yrs = atoms.I(Y)
    n = np.empty((atoms.Nspin, len(atoms.r), atoms.Nstate))
    for spin in range(atoms.Nspin):
        n[spin] = atoms.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin])
    return n


@handle_spin_gracefully
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


def get_grad(scf, spin, W, Y=None, n=None, phi=None, vxc=None):
    '''Calculate the energy gradient with respect to W.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to calculate the gradient for spin up or down.
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
    F = np.diag(atoms.f[spin])
    HW = H(scf, spin, W, Y, n, phi, vxc)
    WHW = W[spin].conj().T @ HW
    # U = Wdag O(W)
    U = W[spin].conj().T @ atoms.O(W[spin])
    invU = inv(U)
    U12 = sqrtm(invU)
    # Htilde = U^-0.5 Wdag H(W) U^-0.5
    Ht = U12 @ WHW @ U12
    # grad E = H(W) - O(W) U^-1 (Wdag H(W)) (U^-0.5 F U^-0.5) + O(W) (U^-0.5 Q(Htilde F - F Htilde))
    return (HW - (atoms.O(W[spin]) @ invU) @ WHW) @ (U12 @ F @ U12) + \
        atoms.O(W[spin]) @ (U12 @ Q(Ht @ F - F @ Ht, U))


def H(scf, spin, W, Y=None, n=None, phi=None, vxc=None):
    '''Left-hand side of the eigenvalue equation.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to calculate the gradient for spin up or down.
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
        n_spin = get_n_spin(atoms, Y, n)
        vxc = get_xc(scf.xc, n_spin, atoms.Nspin)[1]

    # We get the full potential in the functional definition (different to the DFT++ notation)
    # Normally Vxc = Jdag(O(J(exc))) + diag(exc') Jdag(O(J(n)))
    Vxc = atoms.Jdag(atoms.O(atoms.J(vxc[spin])))
    # Vkin = -0.5 L(W)
    Vkin_psi = -0.5 * atoms.L(W[spin])
    # Veff = Jdag(Vion) + Jdag(O(J(vxc))) + Jdag(O(phi))
    Veff = scf.Vloc + Vxc + atoms.Jdag(atoms.O(phi))
    Vnonloc_psi = calc_Vnonloc(scf, W[spin])
    # H = Vkin + Idag(diag(Veff))I + Vnonloc
    return Vkin_psi + atoms.Idag(diagprod(Veff, atoms.I(W[spin]))) + Vnonloc_psi


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
    denom2 = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom2) @ V.conj().T


def get_psi(scf, W, n=None):
    '''Calculate eigenstates from H.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Eigenstates in reciprocal space.
    '''
    atoms = scf.atoms
    Y = orth(atoms, W)
    psi = np.empty_like(Y)
    for spin in range(atoms.Nspin):
        mu = Y[spin].conj().T @ H(scf, spin, W=Y, n=n)
        _, D = eigh(mu)
        psi[spin] = Y[spin] @ D
    return psi


def get_epsilon(scf, W, n=None):
    '''Calculate eigenvalues from H.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Eigenvalues.
    '''
    atoms = scf.atoms
    Y = orth(atoms, W)
    epsilon = np.empty((atoms.Nspin, atoms.Nstate))
    for spin in range(atoms.Nspin):
        mu = Y[spin].conj().T @ H(scf, spin, W=Y, n=n)
        epsilon[spin] = np.sort(eigvalsh(mu))
    return epsilon


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
        W = randn(atoms.Nspin, len(atoms.G2c), atoms.Nstate) + \
            1j * randn(atoms.Nspin, len(atoms.G2c), atoms.Nstate)
    else:
        W = randn(atoms.Nspin, len(atoms.G2c), atoms.Nstate)
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
