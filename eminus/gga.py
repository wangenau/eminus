#!/usr/bin/env python3
'''DFT functions that are only needed for GGA and meta-GGA calculations.'''
import numpy as np


def get_grad_field(atoms, field, real=True):
    '''Calculate the gradient of fields on the grid per spin channel.

    Args:
        atoms: Atoms object.
        field (ndarray): Real-space field per spin channel.

    Keyword Args:
        real (bool): Make the gradient a real array (the gradient of n_spin is real).

    Returns:
        ndarray: Gradients of field per spin channel.
    '''
    dfield = np.empty((atoms.Nspin, len(atoms.r), 3), dtype=complex)
    for spin in range(atoms.Nspin):
        fieldG = atoms.J(field[spin])
        for dim in range(3):
            dfield[spin, :, dim] = atoms.I(1j * atoms.G[:, dim] * fieldG)
    if real:
        return np.real(dfield)
    return dfield


def gradient_correction(atoms, spin, dn_spin, vsigma):
    '''Calculate the gradient correction for the exchange-correlation potential.

    Reference: Chem. Phys. Lett. 199, 557.

    Args:
        atoms: Atoms object.
        spin (int): Spin variable to track weather to calculate the gradient for spin up or down.
        dn_spin (ndarray): Real-space gradient of densities per spin channel.
        vsigma (ndarray): Contracted gradient potential derivative.

    Returns:
        ndarray: Gradient correction in reciprocal space.
    '''
    # sigma is |dn|^2, while vsigma is n * d exc/d sigma
    h = np.zeros_like(dn_spin)
    if atoms.Nspin == 1:
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
    return 1j * np.sum(atoms.G * Gh, axis=1)


def get_tau(atoms, Y):
    '''Calculate the positive-definite kinetic energy densities per spin.

    Reference: J. Chem. Phys. 109, 2092.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        ndarray: Real space positive-definite kinetic energy density.
    '''
    # The "intuitive" way is the one commented out below
    # Sadly, this implementation is really slow for various reasons so use the faster one below

    # Yrs = atoms.I(Y)
    # tau = np.zeros((atoms.Nspin, len(atoms.r)), dtype=complex)
    # for i in range(atoms.Nstate):
    #     dYrs = get_grad_field(atoms, Yrs[..., i], real=False)
    #     tau += 0.5 * atoms.f[:, i, None] * np.sum(dYrs.conj() * dYrs, axis=2)
    # return np.real(tau)

    dYrs = np.empty((atoms.Nspin, len(atoms.r), atoms.Nstate, 3), dtype=complex)
    # Calculate the active G vectors and broadcast to a desired shape
    Gc = atoms.G[atoms.active][None, :, None, :]
    # Calculate the gradients of Y in the active(!) reciprocal space and transform to real space
    for dim in range(3):
        dYrs[..., dim] = atoms.I(1j * Gc[..., dim] * Y)
    # Sum over dimensions (dYx* dYx + dYy* dYy + dYz* dYz)
    sumdYrs = np.real(np.sum(dYrs.conj() * dYrs, axis=3))
    # Sum over all states
    # Use the definition with a division by two
    tau = 0.5 * np.sum(atoms.f[:, None, :] * sumdYrs, axis=2)
    return tau


def calc_Vtau(scf, spin, W, vtau):
    '''Calculate the tau-dependent potential contribution for meta-GGAs.

    Reference: J. Chem. Phys. 145, 204114.

    Args:
        scf: SCF object.
        spin (int): Spin variable to track weather to calculate the gradient for spin up or down.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        vtau (ndarray): Kinetic energy density potential derivative.

    Returns:
        ndarray: Tau-dependent potential contribution in reciprocal space.
    '''
    atoms = scf.atoms

    Vpsi = np.zeros((len(atoms.G2c), atoms.Nstate), dtype=complex)
    if scf.xc_type == 'meta-gga':  # Only calculate the contribution for meta-GGAs
        # The "intuitive" way is the one commented out below
        # Sadly, this implementation is really slow for various reasons so use the faster one below

        # GVpsi = np.empty((len(atoms.G2c), 3), dtype=complex)
        # Gc = atoms.G[atoms.active]
        # Wrs = atoms.I(W)
        # for i in range(atoms.Nstate):
        #     dWrs = get_grad_field(atoms, Wrs[..., i], real=False)
        #     for r in range(3):
        #         GVpsi[:, r] = atoms.J(vtau[spin] * dWrs[spin, :, r], full=False)
        #     Vpsi[:, i] = -0.5 * 1j * np.sum(Gc * GVpsi, axis=1)

        GVpsi = np.empty((len(atoms.G2c), atoms.Nstate, 3), dtype=complex)
        dWrs = np.empty((len(atoms.r), atoms.Nstate, 3), dtype=complex)
        # Calculate the active G vectors and broadcast to a desired shape
        Gc = atoms.G[atoms.active][:, None, :]
        # We only calculate Vtau for one spin channel, index and reshape prior the loop
        vtau_spin = vtau[spin, :, None]
        W_spin = W[spin]
        for dim in range(3):
            # Calculate the gradients of W in the active(!) space and transform to real space
            dWrs[..., dim] = atoms.I(1j * Gc[..., dim] * W_spin)
            # Calculate dexc/dtau * gradpsi and transform to the active reciprocal space
            GVpsi[..., dim] = atoms.J(vtau_spin * dWrs[..., dim], full=False)
        # Sum over dimensions
        # Calculate -0.5 Nabla dot Gvpsi (compare with gradient_correction)
        Vpsi = -0.5 * 1j * np.sum(Gc * GVpsi, axis=2)
    return Vpsi * atoms.Omega
