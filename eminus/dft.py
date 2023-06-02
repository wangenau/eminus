#!/usr/bin/env python3
'''Main DFT functions based on the DFT++ formulation.'''
import numpy as np
from numpy.random import Generator, SFC64
from scipy.linalg import eig, eigh, eigvalsh, inv, sqrtm

from .gga import calc_Vtau, get_grad_field, get_tau, gradient_correction
from .gth import calc_Vnonloc
from .utils import handle_spin_gracefully, pseudo_uniform
from .xc import get_vxc


def solve_poisson(atoms, n):
    '''Solve the Poisson equation.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Hartree field.
    '''
    # phi = -4 pi Linv(O(J(n)))
    return -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))


def get_n_total(atoms, Y, n_spin=None):
    '''Calculate the total electronic density.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n_spin (ndarray): Real-space electronic densities per spin channel.

    Returns:
        ndarray: Electronic density.
    '''
    # Return the total density in the spin-paired case
    if n_spin is not None:
        return np.sum(n_spin, axis=0)

    # n = (IW) F (IW)dag
    Yrs = atoms.I(Y)
    n = np.zeros(len(atoms.r))
    for spin in range(atoms.Nspin):
        n += np.sum(atoms.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin]), axis=1)
    return n


def get_n_spin(atoms, Y, n=None):
    '''Calculate the electronic density per spin channel.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Electronic densities per spin channel.
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

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Orthogonalized wave functions.
    '''
    # Y = W (Wdag O(W))^-0.5
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def get_grad(scf, spin, W, **kwargs):
    '''Calculate the energy gradient with respect to W.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to calculate the gradient for spin up or down.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Gradient.
    '''
    atoms = scf.atoms
    F = np.diag(atoms.f[spin])
    HW = H(scf, spin, W, **kwargs)
    WHW = W[spin].conj().T @ HW
    # U = Wdag O(W)
    OW = atoms.O(W[spin])
    U = W[spin].conj().T @ OW
    invU = inv(U)
    U12 = sqrtm(invU)
    # Htilde = U^-0.5 Wdag H(W) U^-0.5
    Ht = U12 @ WHW @ U12
    # grad E = H(W) - O(W) U^-1 (Wdag H(W)) (U^-0.5 F U^-0.5) + O(W) (U^-0.5 Q(Htilde F - F Htilde))
    return (HW - (OW @ invU) @ WHW) @ (U12 @ F @ U12) + OW @ (U12 @ Q(Ht @ F - F @ Ht, U))


def H(scf, spin, W, dn_spin=None, phi=None, vxc=None, vsigma=None, vtau=None):
    '''Left-hand side of the eigenvalue equation.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to calculate the gradient for spin up or down.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        dn_spin (ndarray): Real-space gradient of densities per spin channel.
        phi (ndarray): Hartree field.
        vxc (ndarray): Exchange-correlation potential.
        vsigma (ndarray): Contracted gradient potential derivative.
        vtau (ndarray): Kinetic energy gradient potential derivative.

    Returns:
        ndarray: Hamiltonian applied on W.
    '''
    atoms = scf.atoms

    # If dn_spin is None all other keyword arguments are None by design
    # In that case precompute values from the SCF class
    if dn_spin is None:
        dn_spin, phi, vxc, vsigma, vtau = H_precompute(scf, W)

    # This calculate the representation in the reciprocal space
    Gvxc = atoms.J(vxc[spin])
    # Calculate the gradient correction to the potential if a GGA functional is used
    if 'gga' in scf.xc_type:
        Gvxc = Gvxc - gradient_correction(atoms, spin, dn_spin, vsigma)
    # We get the full potential in the functional definition (different to the DFT++ notation)
    # Normally Vxc = Jdag(O(J(exc))) + diag(exc') Jdag(O(J(n))) (for LDA functionals)
    Vxc = atoms.Jdag(atoms.O(Gvxc))
    # Vkin = -0.5 L(W)
    Vkin_psi = -0.5 * atoms.L(W[spin])
    # Veff = Jdag(Vion) + Jdag(O(J(vxc))) + Jdag(O(phi))
    Veff = scf.Vloc + Vxc + atoms.Jdag(atoms.O(phi))
    Vnonloc_psi = calc_Vnonloc(scf, W[spin])
    Vtau_psi = calc_Vtau(scf, spin, W, vtau)
    # H = Vkin + Idag(diag(Veff))I + Vnonloc (+ Vtau)
    # Diag(a) * B can be written as a * B if a is a column vector
    return Vkin_psi + atoms.Idag(Veff[:, None] * atoms.I(W[spin])) + Vnonloc_psi + Vtau_psi


def H_precompute(scf, W):
    '''Create precomputed values as intermediate results.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]: dn_spin, phi, vxc, vsigma, and vtau
    '''
    atoms = scf.atoms
    Y = orth(atoms, W)
    n_spin = get_n_spin(atoms, Y)
    n = get_n_total(atoms, Y, n_spin)
    if 'gga' in scf.xc_type:
        dn_spin = get_grad_field(atoms, n_spin)
    else:
        dn_spin = None
    if scf.xc_type == 'meta-gga':
        tau = get_tau(atoms, Y)
    else:
        tau = None
    phi = solve_poisson(atoms, n)
    vxc, vsigma, vtau = get_vxc(scf.xc, n_spin, atoms.Nspin, dn_spin, tau)
    return dn_spin, phi, vxc, vsigma, vtau


def Q(inp, U):
    '''Operator needed to calculate gradients with non-constant occupations.

    Reference: Comput. Phys. Commun. 128, 1.

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


def get_psi(scf, W):
    '''Calculate eigenstates from H.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Eigenstates in reciprocal space.
    '''
    atoms = scf.atoms
    Y = orth(atoms, W)
    psi = np.empty_like(Y)
    for spin in range(atoms.Nspin):
        mu = Y[spin].conj().T @ H(scf, spin, Y)
        _, D = eigh(mu)
        psi[spin] = Y[spin] @ D
    return psi


def get_epsilon(scf, W):
    '''Calculate eigenvalues from H.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Eigenvalues.
    '''
    atoms = scf.atoms
    Y = orth(atoms, W)
    epsilon = np.empty((atoms.Nspin, atoms.Nstate))
    for spin in range(atoms.Nspin):
        mu = Y[spin].conj().T @ H(scf, spin, Y)
        epsilon[spin] = np.sort(eigvalsh(mu))
    return epsilon


def guess_random(scf, complex=True):
    '''Generate random initial-guess coefficients as starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        complex (bool): Use complex numbers for the random guess.

    Returns:
        ndarray: Initial-guess orthogonal wave functions in reciprocal space.
    '''
    atoms = scf.atoms

    seed = 42
    rng = Generator(SFC64(seed))
    if complex:
        W = rng.standard_normal((atoms.Nspin, len(atoms.G2c), atoms.Nstate)) + \
            1j * rng.standard_normal((atoms.Nspin, len(atoms.G2c), atoms.Nstate))
    else:
        W = rng.standard_normal((atoms.Nspin, len(atoms.G2c), atoms.Nstate))
    return orth(atoms, W)


def guess_pseudo(scf, seed=1234):
    '''Generate initial-guess coefficients using pseudo-random starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        seed (int): Seed to initialize the random number generator.

    Returns:
        ndarray: Initial-guess orthogonal wave functions in reciprocal space.
    '''
    atoms = scf.atoms
    W = pseudo_uniform((atoms.Nspin, len(atoms.G2c), atoms.Nstate), seed=seed)
    return orth(atoms, W)
