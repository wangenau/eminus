# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""DFT functions that are only needed for (meta-)GGA calculations.

Most functions here have been optimized for performance and can be harder to understand than normal.
To mitigate this, easier (but slower) implementations have been added as comments.
"""

import numpy as np

from .utils import handle_k


def get_grad_field(atoms, field, real=True):
    """Calculate the gradient of fields on the grid per spin channel.

    Args:
        atoms: Atoms object.
        field: Real-space field per spin channel.

    Keyword Args:
        real: Make the gradient a real array (the gradient of n_spin is real).

    Returns:
        Gradients of field per spin channel.
    """
    dfield = np.empty((atoms.occ.Nspin, atoms.Ns, 3), dtype=complex)
    for spin in range(atoms.occ.Nspin):
        fieldG = atoms.J(field[spin])
        for dim in range(3):
            dfield[spin, :, dim] = atoms.I(1j * atoms.G[:, dim] * fieldG)
    if real:
        return np.real(dfield)
    return dfield


def gradient_correction(atoms, spin, dn_spin, vsigma):
    """Calculate the gradient correction for the exchange-correlation potential.

    Reference: Chem. Phys. Lett. 199, 557.

    Args:
        atoms: Atoms object.
        spin: Spin variable to track whether to do the calculation for spin up or down.
        dn_spin: Real-space gradient of densities per spin channel.
        vsigma: Contracted gradient potential derivative.

    Returns:
        Gradient correction in reciprocal space.
    """
    # sigma is |dn|^2, while vsigma is n * d exc/d sigma
    h = np.empty_like(dn_spin)
    if not atoms.unrestricted:
        # In the unpolarized case we have no spin mixing and only one spin density
        h[0] = 2 * vsigma[0, :, None] * dn_spin[0]
    else:
        # In the polarized case we would get for spin up (and similar for spin down)
        # Vxc_u = vxc_u - Nabla dot (2 vsigma_uu * dn_u + vsigma_ud * dn_d)
        # h is the expression in the brackets
        h[0] = 2 * vsigma[0, :, None] * dn_spin[0] + vsigma[1, :, None] * dn_spin[1]
        h[1] = 2 * vsigma[2, :, None] * dn_spin[1] + vsigma[1, :, None] * dn_spin[0]

    # Calculate Nabla dot h
    # Normally we would calculate the correction for each spin, but we only need one at a time in H
    Gh = np.empty((len(atoms.G2), 3), dtype=complex)
    for dim in range(3):
        Gh[:, dim] = atoms.J(h[spin, :, dim])
    # return 1j * np.sum(atoms.G * Gh, axis=1)
    return 1j * np.einsum("ir,ir->i", atoms.G, Gh)


@handle_k(mode="reduce")
def get_tau(atoms, Y, ik):
    """Calculate the positive-definite kinetic energy densities per spin.

    Reference: J. Chem. Phys. 109, 2092.

    Args:
        atoms: Atoms object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.
        ik: k-point index.

    Returns:
        Real-space positive-definite kinetic energy density.
    """
    # The "intuitive" way is the one commented out below (without k-point dependency)
    # Sadly, this implementation is really slow for various reasons so use the faster one below

    # Yrs = atoms.I(Y)
    # tau = np.zeros((atoms.occ.Nspin, atoms.Ns), dtype=complex)
    # for i in range(atoms.occ.Nstate):
    #     dYrs = get_grad_field(atoms, Yrs[..., i], real=False)
    #     tau += 0.5 * atoms.occ.f[:, i, None] * np.sum(dYrs.conj() * dYrs, axis=2)
    # return np.real(tau)

    dYrs = np.empty((atoms.occ.Nspin, atoms.Ns, Y.shape[-1], 3), dtype=complex)
    # Calculate the active G vectors and broadcast to the needed shape
    Gkc = atoms.G[atoms.active[ik]][:, None, :] + atoms.kpts.k[ik]
    # Calculate the gradients of Y in the active(!) reciprocal space and transform to real-space
    for dim in range(3):
        dYrs[..., dim] = atoms.I(1j * Gkc[..., dim] * Y, ik)
    # Sum over dimensions (dYx* dYx + dYy* dYy + dYz* dYz)
    # sumdYrs = np.real(np.sum(dYrs.conj() * dYrs, axis=3))
    # Sum over all states
    # Use the definition with a division by two
    # return 0.5 * np.sum(atoms.occ.f[:, None, :] * sumdYrs, axis=2)
    # Or in compressed Einstein notation:
    return (
        0.5
        * atoms.kpts.wk[ik]
        * np.real(
            np.einsum("sj,sijr,sijr->si", atoms.occ.f[ik], dYrs.conj(), dYrs, optimize="greedy")
        )
    )


def calc_Vtau(scf, ik, spin, W, vtau):
    """Calculate the tau-dependent potential contribution for meta-GGAs.

    Reference: J. Chem. Phys. 145, 204114.

    Args:
        scf: SCF object.
        ik: k-point index.
        spin: Spin variable to track whether to do the calculation for spin up or down.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        vtau: Kinetic energy density potential derivative.

    Returns:
        Tau-dependent potential contribution in reciprocal space.
    """
    atoms = scf.atoms

    if scf.xc_type != "meta-gga":  # Only calculate the contribution for meta-GGAs
        return np.zeros((len(atoms.Gk2c[ik]), W[ik].shape[-1]), dtype=complex)

    # The "intuitive" way is the one commented out below (without k-point dependency)
    # Sadly, this implementation is really slow for various reasons so use the faster one below

    # GVpsi = np.empty((len(atoms.G2c), 3), dtype=complex)
    # Gc = atoms.G[atoms.active]
    # Wrs = atoms.I(W)
    # for i in range(atoms.occ.Nstate):
    #     dWrs = get_grad_field(atoms, Wrs[..., i], real=False)
    #     for r in range(3):
    #         GVpsi[:, r] = atoms.J(vtau[spin] * dWrs[spin, :, r], full=False)
    #     Vpsi[:, i] = -0.5 * 1j * np.sum(Gc * GVpsi, axis=1)

    GVpsi = np.empty((len(atoms.Gk2c[ik]), W[ik].shape[-1], 3), dtype=complex)
    dWrs = np.empty((atoms.Ns, W[ik].shape[-1], 3), dtype=complex)
    # Calculate the active G vectors and broadcast to the needed shape
    Gkc = atoms.G[atoms.active[ik]][:, None, :] + scf.kpts.k[ik]
    # We only calculate Vtau for one spin channel, index, and reshape before the loop
    vtau_spin = vtau[spin, :, None]
    W_spin = W[ik][spin]
    for dim in range(3):
        # Calculate the gradients of W in the active(!) space and transform to real-space
        dWrs[..., dim] = atoms.I(1j * Gkc[..., dim] * W_spin, ik)
        # Calculate dexc/dtau * gradpsi and transform to the active reciprocal space
        GVpsi[..., dim] = atoms.J(vtau_spin * dWrs[..., dim], ik, full=False)
    # Sum over dimensions
    # Calculate -0.5 Nabla dot Gvpsi (compare with gradient_correction)
    # Vpsi = -0.5 * 1j * np.sum(Gkc * GVpsi, axis=2)
    Vpsi = -0.5 * 1j * np.einsum("ior,ijr->ij", Gkc, GVpsi)
    return atoms.O(Vpsi)
