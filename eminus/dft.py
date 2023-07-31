#!/usr/bin/env python3
"""Main DFT functions based on the DFT++ formulation."""
import numpy as np
from numpy.random import Generator, SFC64
from scipy.linalg import eig, eigh, eigvalsh, inv, sqrtm

from .gga import calc_Vtau, get_grad_field, get_tau, gradient_correction
from .gth import calc_Vnonloc
from .utils import handle_spin_gracefully, pseudo_uniform
from .xc import get_vxc


def solve_poisson(atoms, n):
    """Solve the Poisson equation.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n (ndarray): Real-space electronic density.

    Returns:
        ndarray: Hartree field.
    """
    # phi = -4 pi Linv(O(J(n)))
    return -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))


def get_n_total(atoms, Y, n_spin=None):
    """Calculate the total electronic density.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n_spin (ndarray): Real-space electronic densities per spin channel.

    Returns:
        ndarray: Electronic density.
    """
    # Return the total density in the spin-paired case
    if n_spin is not None:
        return np.sum(n_spin, axis=0)

    # n = (IW) F (IW)dag
    Yrs = atoms.I(Y)
    n = np.zeros(len(atoms.r))
    for spin in range(atoms.occ.Nspin):
        n += np.sum(atoms.occ.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin]), axis=1)
    return n


def get_n_spin(atoms, Y):
    """Calculate the electronic density per spin channel.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        ndarray: Electronic densities per spin channel.
    """
    Yrs = atoms.I(Y)
    n = np.empty((atoms.occ.Nspin, len(atoms.r)))
    for spin in range(atoms.occ.Nspin):
        n[spin] = np.sum(atoms.occ.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin]), axis=1)
    return n


def get_n_single(atoms, Y):
    """Calculate the single-electron densities.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        ndarray: Single-electron densities.
    """
    Yrs = atoms.I(Y)
    n = np.empty((atoms.occ.Nspin, len(atoms.r), atoms.occ.Nstate))
    for spin in range(atoms.occ.Nspin):
        n[spin] = atoms.occ.f[spin] * np.real(Yrs[spin].conj() * Yrs[spin])
    return n


@handle_spin_gracefully
def orth(atoms, W):
    """Orthogonalize coefficient matrix W.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Orthogonalized wave functions.
    """
    # Y = W (Wdag O(W))^-0.5
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def get_grad(scf, spin, W, **kwargs):
    """Calculate the energy gradient with respect to W.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to do the calculation for spin up or down.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        ndarray: Gradient.
    """
    atoms = scf.atoms
    F = atoms.occ.F[spin]
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
    """Left-hand side of the eigenvalue equation.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to do the calculation for spin up or down.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        dn_spin (ndarray): Real-space gradient of densities per spin channel.
        phi (ndarray): Hartree field.
        vxc (ndarray): Exchange-correlation potential.
        vsigma (ndarray): Contracted gradient potential derivative.
        vtau (ndarray): Kinetic energy gradient potential derivative.

    Returns:
        ndarray: Hamiltonian applied on W.
    """
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
    # Vkin = -0.5 L(W)
    Vkin_psi = -0.5 * atoms.L(W[spin])
    # Veff = Jdag(Vion) + Jdag(O(J(vxc))) + Jdag(O(phi))
    # We get the full potential in the functional definition (different to the DFT++ notation)
    # Normally Vxc = Jdag(O(J(exc))) + diag(exc') Jdag(O(J(n))) (for LDA functionals)
    Veff = scf.Vloc + atoms.Jdag(atoms.O(Gvxc + phi))
    Vnonloc_psi = calc_Vnonloc(scf, spin, W)
    Vtau_psi = calc_Vtau(scf, spin, W, vtau)
    # H = Vkin + Idag(diag(Veff))I + Vnonloc (+ Vtau)
    # Diag(a) * B can be written as a * B if a is a column vector
    return Vkin_psi + atoms.Idag(Veff[:, None] * atoms.I(W[spin])) + Vnonloc_psi + Vtau_psi


def H_precompute(scf, W):
    """Create precomputed values as intermediate results.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]: dn_spin, phi, vxc, vsigma, and vtau
    """
    # We are calculating everything here over both spin channels
    # In theory we would be fine/faster by only calculating the currently used spin channel
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
    vxc, vsigma, vtau = get_vxc(scf.xc, n_spin, atoms.occ.Nspin, dn_spin, tau)
    return dn_spin, phi, vxc, vsigma, vtau


def Q(inp, U):
    """Operator needed to calculate gradients with non-constant occupations.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        inp (ndarray): Coefficients input array.
        U (ndarray): Overlap of wave functions.

    Returns:
        ndarray: Q operator result.
    """
    mu, V = eig(U)
    mu = mu[:, None]
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom2 = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom2) @ V.conj().T


def get_psi(scf, W):
    """Calculate eigenstates from H.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Eigenstates in reciprocal space.
    """
    atoms = scf.atoms
    Y = orth(atoms, W)
    psi = np.empty_like(Y)
    for spin in range(atoms.occ.Nspin):
        mu = Y[spin].conj().T @ H(scf, spin, Y)
        _, D = eigh(mu)
        psi[spin] = Y[spin] @ D
    return psi


def get_epsilon(scf, W):
    """Calculate eigenvalues from H.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: Eigenvalues.
    """
    atoms = scf.atoms
    Y = orth(atoms, W)
    epsilon = np.empty((atoms.occ.Nspin, atoms.occ.Nstate))
    for spin in range(atoms.occ.Nspin):
        mu = Y[spin].conj().T @ H(scf, spin, Y)
        epsilon[spin] = np.sort(eigvalsh(mu))
    return epsilon


def guess_random(scf, seed=42, symmetric=False):
    """Generate random initial-guess coefficients as starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        seed (int): Seed to initialize the random number generator.
        symmetric (bool): Weather to use the same guess for both spin channels.

    Returns:
        ndarray: Initial-guess orthogonal wave functions in reciprocal space.
    """
    atoms = scf.atoms
    rng = Generator(SFC64(seed))
    if symmetric:
        W = rng.standard_normal((len(atoms.G2c), atoms.occ.Nstate)) + \
            1j * rng.standard_normal((len(atoms.G2c), atoms.occ.Nstate))
        W = np.array([W] * atoms.occ.Nspin)
    else:
        W = rng.standard_normal((atoms.occ.Nspin, len(atoms.G2c), atoms.occ.Nstate)) + \
            1j * rng.standard_normal((atoms.occ.Nspin, len(atoms.G2c), atoms.occ.Nstate))
    return orth(atoms, W)


def guess_pseudo(scf, seed=1234, symmetric=False):
    """Generate initial-guess coefficients using pseudo-random starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        seed (int): Seed to initialize the random number generator.
        symmetric (bool): Weather to use the same guess for both spin channels.

    Returns:
        ndarray: Initial-guess orthogonal wave functions in reciprocal space.
    """
    atoms = scf.atoms
    if symmetric:
        W = pseudo_uniform((1, len(atoms.G2c), atoms.occ.Nstate), seed=seed)
        W = np.array([W[0]] * atoms.occ.Nspin)
    else:
        W = pseudo_uniform((atoms.occ.Nspin, len(atoms.G2c), atoms.occ.Nstate), seed=seed)
    return orth(atoms, W)
