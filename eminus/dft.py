# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Main DFT functions based on the DFT++ formulation."""

import numpy as np
from numpy.linalg import multi_dot
from numpy.random import Generator, SFC64
from scipy.linalg import eig, eigh, eigvalsh, inv, sqrtm

from .gga import calc_Vtau, get_grad_field, get_tau, gradient_correction
from .gth import calc_Vnonloc
from .logger import log
from .utils import handle_k, handle_spin, pseudo_uniform
from .xc import get_vxc


def get_phi(atoms, n):
    """Solve the Poisson equation.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n: Real-space electronic density.

    Returns:
        Hartree field.
    """
    # phi = -4 pi Linv(O(J(n)))
    return -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))


@handle_k
@handle_spin
def orth(atoms, W):
    """Orthogonalize coefficient matrix W.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        Orthogonalized wave functions.
    """
    # Y = W (Wdag O(W))^-0.5
    return W @ inv(sqrtm(W.conj().T @ atoms.O(W)))


def orth_unocc(atoms, Y, Z):
    """Orthogonalize unoccupied matrix Z while maintaining orthogonality to Y.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.
        Z: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        Orthogonalized wave functions.
    """
    D = [np.empty_like(Zk) for Zk in Z]
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            # rhoZ = (I - Y Ydag O) Z
            Yocc = Y[ik][spin][:, atoms.occ.f[ik][spin] > 0]
            rhoZ = Z[ik][spin] - Yocc @ Yocc.conj().T @ atoms.O(Z[ik][spin])
            # D = rhoZ (rhoZdag O(rhoZ))^-0.5
            D[ik][spin] = rhoZ @ inv(sqrtm(rhoZ.conj().T @ atoms.O(rhoZ)))
    return D


def get_n_total(atoms, Y, n_spin=None):
    """Calculate the total electronic density.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n_spin: Real-space electronic densities per spin channel.

    Returns:
        Electronic density.
    """
    # Return the total density in the spin-paired case
    if n_spin is not None:
        return np.sum(n_spin, axis=0)

    # n = (IW) F (IW)dag
    n = np.zeros(atoms.Ns)
    Yrs = atoms.I(Y)
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            n += np.sum(
                atoms.occ.f[ik, spin]
                * atoms.kpts.wk[ik]
                * np.real(Yrs[ik][spin].conj() * Yrs[ik][spin]),
                axis=1,
            )
    return n


@handle_k(mode="reduce")
def get_n_spin(atoms, Y, ik):
    """Calculate the electronic density per spin channel.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.
        ik: k-point index.

    Returns:
        Electronic densities per spin channel.
    """
    Yrs = atoms.I(Y, ik)
    n = np.empty((atoms.occ.Nspin, atoms.Ns))
    for spin in range(atoms.occ.Nspin):
        n[spin] = np.sum(
            atoms.occ.f[ik, spin] * atoms.kpts.wk[ik] * np.real(Yrs[spin].conj() * Yrs[spin]),
            axis=1,
        )
    return n


@handle_k(mode="reduce")
def get_n_single(atoms, Y, ik):
    """Calculate the single-electron densities.

    Args:
        atoms: Atoms object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.
        ik: k-point index.

    Returns:
        Single-electron densities.
    """
    Yrs = atoms.I(Y, ik)
    n = np.empty((atoms.occ.Nspin, atoms.Ns, atoms.occ.Nstate))
    for spin in range(atoms.occ.Nspin):
        n[spin] = atoms.occ.f[ik, spin] * atoms.kpts.wk[ik] * np.real(Yrs[spin].conj() * Yrs[spin])
    return n


def get_grad(scf, ik, spin, W, **kwargs):
    """Calculate the energy gradient with respect to W.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        ik: k-point index.
        spin: Spin variable to track whether to do the calculation for spin up or down.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        Gradient.
    """
    atoms = scf.atoms
    F = atoms.occ.F[ik][spin]
    HW = H(scf, ik, spin, W, **kwargs)
    WHW = W[ik][spin].conj().T @ HW
    # U = Wdag O(W)
    OW = atoms.O(W[ik][spin])
    U = W[ik][spin].conj().T @ OW
    invU = inv(U)
    U12 = sqrtm(invU)
    # Htilde = U^-0.5 Wdag H(W) U^-0.5
    Ht = multi_dot([U12, WHW, U12])
    # grad E = H(W) - O(W) U^-1 (Wdag H(W)) (U^-0.5 F U^-0.5) + O(W) (U^-0.5 Q(Htilde F - F Htilde))
    tmp = multi_dot([OW, invU, WHW])
    return atoms.kpts.wk[ik] * (
        multi_dot([HW - tmp, U12, F, U12]) + multi_dot([OW, U12, Q(Ht @ F - F @ Ht, U)])
    )


def H(scf, ik, spin, W, dn_spin=None, phi=None, vxc=None, vsigma=None, vtau=None):
    """Left-hand side of the eigenvalue equation.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        ik: k-point index.
        spin: Spin variable to track whether to do the calculation for spin up or down.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        phi: Hartree field.
        vxc: Exchange-correlation potential.
        vsigma: Contracted gradient potential derivative.
        vtau: Kinetic energy gradient potential derivative.

    Returns:
        Hamiltonian applied on W.
    """
    atoms = scf.atoms

    # If dn_spin is None all other keyword arguments are None by design
    # In that case precompute values from the SCF class
    if phi is None:
        dn_spin, phi, vxc, vsigma, vtau = H_precompute(scf, W)

    # This calculates the XC potential in the reciprocal space
    Gvxc = atoms.J(vxc[spin])
    # Calculate the gradient correction to the potential if a (meta-)GGA functional is used
    if "gga" in scf.xc_type:
        Gvxc = Gvxc - gradient_correction(atoms, spin, dn_spin, vsigma)
    # Vkin = -0.5 L(W)
    Vkin_psi = -0.5 * atoms.L(W[ik][spin], ik)
    # Veff = Jdag(Vion) + Jdag(O(J(vxc))) + Jdag(O(phi))
    # We get the full potential in the functional definition (different to the DFT++ notation)
    # Normally Vxc = Jdag(O(J(exc))) + diag(exc') Jdag(O(J(n))) (for LDA functionals)
    Veff = scf.Vloc + atoms.Jdag(atoms.O(Gvxc + phi))
    Vnonloc_psi = calc_Vnonloc(scf, ik, spin, W)
    Vtau_psi = calc_Vtau(scf, ik, spin, W, vtau)
    # H = Vkin + Idag(diag(Veff))I + Vnonloc (+ Vtau)
    # Diag(a) * B can be written as a * B if a is a column vector
    return (
        Vkin_psi + atoms.Idag(Veff[:, None] * atoms.I(W[ik][spin], ik), ik) + Vnonloc_psi + Vtau_psi
    )


def H_precompute(scf, W):
    """Create precomputed values as intermediate results.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        dn_spin, phi, vxc, vsigma, and vtau.
    """
    # We are calculating everything here over both spin channels
    # We would be fine/could improve performance by only calculating the currently used spin channel
    atoms = scf.atoms
    Y = orth(atoms, W)
    n_spin = get_n_spin(atoms, Y)
    n = get_n_total(atoms, Y, n_spin)
    if "gga" in scf.xc_type:
        dn_spin = get_grad_field(atoms, n_spin)
    else:
        dn_spin = None
    if scf.xc_type == "meta-gga":
        tau = get_tau(atoms, Y)
    else:
        tau = None
    phi = get_phi(atoms, n)
    vxc, vsigma, vtau = get_vxc(scf.xc, n_spin, atoms.occ.Nspin, dn_spin, tau, scf.xc_params)
    return dn_spin, phi, vxc, vsigma, vtau


def Q(inp, U):
    """Operator needed to calculate gradients with non-constant occupations.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        inp: Coefficients input array.
        U: Overlap of wave functions.

    Returns:
        Q operator result.
    """
    mu, V = eig(U)
    mu = mu[:, None]
    denom = np.sqrt(mu) @ np.ones((1, len(mu)))
    denom2 = denom + denom.conj().T
    # return V @ ((V.conj().T @ inp @ V) / denom2) @ V.conj().T
    tmp = multi_dot([V.conj().T, inp, V])
    return multi_dot([V, tmp / denom2, V.conj().T])


def get_psi(scf, W, **kwargs):
    """Calculate eigenstates from H.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        Eigenstates in reciprocal space.
    """
    if W is None:
        log.warning('The provided wave function is "None".')
        return None

    atoms = scf.atoms
    Y = orth(atoms, W)
    psi = [np.empty_like(Yk) for Yk in Y]
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            mu = Y[ik][spin].conj().T @ H(scf, ik, spin, Y, **kwargs)
            _, D = eigh(mu)
            psi[ik][spin] = Y[ik][spin] @ D
    return psi


def get_epsilon(scf, W, **kwargs):
    """Calculate eigenvalues from H.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        Eigenvalues.
    """
    if W is None:
        log.warning('The provided wave function is "None".')
        return None

    atoms = scf.atoms
    Y = orth(atoms, W)
    epsilon = np.empty((atoms.kpts.Nk, atoms.occ.Nspin, Y[0].shape[-1]))
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            mu = Y[ik][spin].conj().T @ H(scf, ik, spin, Y, **kwargs)
            epsilon[ik][spin] = np.sort(eigvalsh(mu))
    return epsilon


def get_epsilon_unocc(scf, W, Z, **kwargs):
    """Calculate eigenvalues from H of unoccupied states.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Z: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        Eigenvalues.
    """
    if W is None or Z is None:
        log.warning('The provided wave function is "None".')
        return None

    atoms = scf.atoms
    Y = orth(atoms, W)
    D = orth_unocc(atoms, Y, Z)
    epsilon = np.empty((atoms.kpts.Nk, atoms.occ.Nspin, D[0].shape[-1]))
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            mu = D[ik][spin].conj().T @ H(scf, ik, spin, D, **kwargs)
            epsilon[ik][spin] = np.sort(eigvalsh(mu))
    return epsilon


def guess_random(scf, Nstate=None, seed=42, symmetric=False):
    """Generate random initial-guess coefficients as starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        Nstate: Number of states.
        seed: Seed to initialize the random number generator.
        symmetric: Whether to use the same guess for both spin channels.

    Returns:
        Initial-guess orthogonal wave functions in reciprocal space.
    """
    atoms = scf.atoms
    if Nstate is None:
        Nstate = atoms.occ.Nstate

    rng = Generator(SFC64(seed))
    W = []
    for ik in range(atoms.kpts.Nk):
        if symmetric:
            W_ik = rng.standard_normal((len(atoms.Gk2c[ik]), Nstate)) + 1j * rng.standard_normal(
                (len(atoms.Gk2c[ik]), Nstate)
            )
            W.append(np.array([W_ik] * atoms.occ.Nspin))
        else:
            W_ik = rng.standard_normal(
                (atoms.occ.Nspin, len(atoms.Gk2c[ik]), Nstate)
            ) + 1j * rng.standard_normal((atoms.occ.Nspin, len(atoms.Gk2c[ik]), Nstate))
            W.append(W_ik)
    return orth(atoms, W)


def guess_pseudo(scf, Nstate=None, seed=1234, symmetric=False):
    """Generate initial-guess coefficients using pseudo-random starting values.

    Args:
        scf: SCF object.

    Keyword Args:
        Nstate: Number of states.
        seed: Seed to initialize the random number generator.
        symmetric: Whether to use the same guess for both spin channels.

    Returns:
        Initial-guess orthogonal wave functions in reciprocal space.
    """
    atoms = scf.atoms
    if Nstate is None:
        Nstate = atoms.occ.Nstate

    W = []
    for ik in range(atoms.kpts.Nk):
        if symmetric:
            W_ik = pseudo_uniform((1, len(atoms.Gk2c[ik]), Nstate), seed=seed)
            W.append(np.array([W_ik[0]] * atoms.occ.Nspin))
        else:
            W_ik = pseudo_uniform((atoms.occ.Nspin, len(atoms.Gk2c[ik]), Nstate), seed=seed)
            W.append(W_ik)
    return orth(atoms, W)
