# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Minimization algorithms for fixed Hamiltonians.

Similar to :mod:`eminus.minimizer` but for a fixed Hamiltonian the implementation can be simplified
and made more performant.
"""

import copy
import logging

import numpy as np
from scipy.linalg import inv, sqrtm

from .dft import H, orth, orth_unocc
from .energies import get_Eband
from .logger import name
from .minimizer import cg_method, cg_test, check_convergence, linmin_test
from .utils import dotprod


def scf_step_occ(scf, W):
    """Perform one SCF step for an occupied band minimization calculation.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        Band energy.
    """
    atoms = scf.atoms
    scf.Y = orth(atoms, W)
    return get_Eband(scf, scf.Y, **scf._precomputed)


def scf_step_unocc(scf, Z):
    """Perform one SCF step for an unoccupied band minimization calculation.

    Args:
        scf: SCF object.
        Z: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        Band energy.
    """
    atoms = scf.atoms
    scf.D = orth_unocc(atoms, scf.Y, Z)
    return get_Eband(scf, scf.D, **scf._precomputed)


def get_grad_occ(scf, ik, spin, W, **kwargs):
    """Calculate the occupied band energy gradient with respect to W.

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
    W = orth(atoms, W)
    HW = H(scf, ik, spin, W, **kwargs)
    WHW = W[ik][spin].conj().T @ HW
    OW = atoms.O(W[ik][spin])
    U = W[ik][spin].conj().T @ OW
    invU = inv(U)
    U12 = sqrtm(invU)
    # grad E = (I - O(Y) Ydag) H(Y) U^-0.5
    return atoms.kpts.wk[ik] * ((HW - OW @ WHW) @ U12)


def get_grad_unocc(scf, ik, spin, Z, **kwargs):
    """Calculate the unoccupied band energy gradient with respect to Z.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        ik: k-point index.
        spin: Spin variable to track whether to do the calculation for spin up or down.
        Z: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        Gradient.
    """
    atoms = scf.atoms
    Y = scf.Y[ik][spin][:, scf.atoms.occ.f[ik][spin] > 0]
    Ydag = Y.conj().T
    # We need X12 later, so orthogonalize in-place and only the current state
    rhoZ = Z[ik][spin] - Y @ Ydag @ atoms.O(Z[ik][spin])
    X12 = inv(sqrtm(rhoZ.conj().T @ atoms.O(rhoZ)))
    D = rhoZ @ X12
    # Create the correct input shape for the Hamiltonian
    D_tmp = [None] * len(Z)
    D_tmp[ik] = np.empty_like(Z[ik])
    D_tmp[ik][spin] = D
    HD = H(scf, ik, spin, D_tmp, **kwargs)
    DHD = D.conj().T @ HD
    I = np.eye(Z[ik].shape[1])
    # grad E = (I - O(Y) Ydag) (I - O(D) Ddag) H(D) X^-0.5
    return atoms.kpts.wk[ik] * ((I - atoms.O(Y) @ Ydag) @ (HD - atoms.O(D) @ DHD) @ X12)


@name("steepest descent minimization")
def sd(
    scf,
    W,
    Nit,
    cost=scf_step_occ,
    grad=get_grad_occ,
    condition=check_convergence,
    betat=3e-5,
    **kwargs,
):
    """Steepest descent minimization algorithm for a fixed Hamiltonian.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        **kwargs: Throwaway arguments.

    Returns:
        Band energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    for _ in range(Nit):
        c = cost(scf, W)
        costs.append(c)
        if condition(scf, "sd", costs):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, W, **scf._precomputed)
                W[ik][spin] = W[ik][spin] - betat * g
    return costs, W


@name("preconditioned line minimization")
def pclm(
    scf,
    W,
    Nit,
    cost=scf_step_occ,
    grad=get_grad_occ,
    condition=check_convergence,
    betat=3e-5,
    precondition=True,
    **kwargs,
):
    """Preconditioned line minimization algorithm for a fixed Hamiltonian.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        precondition: Whether to use a preconditioner.
        **kwargs: Throwaway arguments.

    Returns:
        Band energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    linmin = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    d = [np.empty_like(Wk) for Wk in W]

    if precondition:
        method = "pclm"
    else:
        method = "lm"

    for i in range(Nit):
        c = cost(scf, W)
        costs.append(c)
        if condition(scf, method, costs, linmin):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, W, **scf._precomputed)
                if scf._log.level <= logging.DEBUG and i > 0:
                    linmin[ik][spin] = linmin_test(g, d[ik][spin])
                if precondition:
                    d[ik][spin] = -atoms.K(g, ik)
                else:
                    d[ik][spin] = -g
                W[ik][spin] = W[ik][spin] + betat * d[ik][spin]
                gt = grad(scf, ik, spin, W, **scf._precomputed)
                beta = abs(betat * dotprod(g, d[ik][spin]) / dotprod(g - gt, d[ik][spin]))
                W[ik][spin] = W[ik][spin] + beta * d[ik][spin]
    return costs, W


@name("line minimization")
def lm(
    scf,
    W,
    Nit,
    cost=scf_step_occ,
    grad=get_grad_occ,
    condition=check_convergence,
    betat=3e-5,
    **kwargs,
):
    """Line minimization algorithm for a fixed Hamiltonian.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        **kwargs: Throwaway arguments.

    Returns:
        Band energies per SCF cycle and optimized expansion coefficients.
    """
    return pclm(scf, W, Nit, cost, grad, condition, betat, precondition=False)


@name("preconditioned conjugate-gradient minimization")
def pccg(
    scf,
    W,
    Nit,
    cost=scf_step_occ,
    grad=get_grad_occ,
    condition=check_convergence,
    betat=3e-5,
    cgform=1,
    precondition=True,
):
    """Preconditioned conjugate-gradient minimization algorithm for a fixed Hamiltonian.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        cgform: Conjugate gradient form.
        precondition: Whether to use a preconditioner.

    Returns:
        Band energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    linmin = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    cg = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    norm_g = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    d = [np.empty_like(Wk) for Wk in W]
    d_old = [np.empty_like(Wk) for Wk in W]
    g_old = [np.empty_like(Wk) for Wk in W]

    if precondition:
        method = "pccg"
    else:
        method = "cg"

    c = cost(scf, W)
    costs.append(c)
    condition(scf, method, costs)
    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            g = grad(scf, ik, spin, W, **scf._precomputed)
            if precondition:
                d[ik][spin] = -atoms.K(g, ik)
            else:
                d[ik][spin] = -g
            W[ik][spin] = W[ik][spin] + betat * d[ik][spin]
            gt = grad(scf, ik, spin, W, **scf._precomputed)
            beta = abs(betat * dotprod(g, d[ik][spin]) / dotprod(g - gt, d[ik][spin]))
            g_old[ik][spin], d_old[ik][spin] = g, d[ik][spin]
            W[ik][spin] = W[ik][spin] + beta * d[ik][spin]

    for _ in range(1, Nit):
        c = cost(scf, W)
        costs.append(c)
        if condition(scf, method, costs, linmin, cg, norm_g):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf._log.level <= logging.DEBUG:
                    linmin[ik][spin] = linmin_test(g, d[ik][spin])
                    cg[ik][spin] = cg_test(atoms, ik, g, g_old[ik][spin], precondition)
                beta, norm_g[ik][spin] = cg_method(
                    scf, ik, cgform, g, g_old[ik][spin], d_old[ik][spin], precondition
                )
                if precondition:
                    d[ik][spin] = -atoms.K(g, ik) + beta * d_old[ik][spin]
                else:
                    d[ik][spin] = -g + beta * d_old[ik][spin]
                W[ik][spin] = W[ik][spin] + betat * d[ik][spin]
                gt = grad(scf, ik, spin, W, **scf._precomputed)
                beta = abs(betat * dotprod(g, d[ik][spin]) / dotprod(g - gt, d[ik][spin]))
                g_old[ik][spin], d_old[ik][spin] = g, d[ik][spin]
                W[ik][spin] = W[ik][spin] + beta * d[ik][spin]
    return costs, W


@name("conjugate-gradient minimization")
def cg(
    scf,
    W,
    Nit,
    cost=scf_step_occ,
    grad=get_grad_occ,
    condition=check_convergence,
    betat=3e-5,
    cgform=1,
):
    """Conjugate-gradient minimization algorithm for a fixed Hamiltonian.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        cgform: Conjugate gradient form.

    Returns:
        Band energies per SCF cycle and optimized expansion coefficients.
    """
    return pccg(scf, W, Nit, cost, grad, condition, betat, cgform, precondition=False)


@name("auto minimization")
def auto(
    scf,
    W,
    Nit,
    cost=scf_step_occ,
    grad=get_grad_occ,
    condition=check_convergence,
    betat=3e-5,
    cgform=1,
):
    """Automatic precond. conjugate-gradient minimization algorithm for a fixed Hamiltonian.

    This function chooses an sd step over the pccg step if the energy goes up.

    Args:
        scf: SCF object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        cgform: Conjugate gradient form.

    Returns:
        Band energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    linmin = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    cg = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    norm_g = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    g = [np.empty_like(Wk) for Wk in W]
    d = [np.empty_like(Wk) for Wk in W]
    d_old = [np.empty_like(Wk) for Wk in W]
    g_old = [np.empty_like(Wk) for Wk in W]

    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            g[ik][spin] = grad(scf, ik, spin, W, **scf._precomputed)
            d[ik][spin] = -atoms.K(g[ik][spin], ik)
            W[ik][spin] = W[ik][spin] + betat * d[ik][spin]
            gt = grad(scf, ik, spin, W, **scf._precomputed)
            beta = abs(
                betat * dotprod(g[ik][spin], d[ik][spin]) / dotprod(g[ik][spin] - gt, d[ik][spin])
            )
            g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]
            W[ik][spin] = W[ik][spin] + beta * d[ik][spin]

    c = cost(scf, W)
    costs.append(c)
    if condition(scf, "pccg", costs):
        return costs

    for _ in range(1, Nit):
        W_old = copy.deepcopy(W)
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf._log.level <= logging.DEBUG:
                    linmin[ik][spin] = linmin_test(g[ik][spin], d[ik][spin])
                    cg[ik][spin] = cg_test(atoms, ik, g[ik][spin], g_old[ik][spin])
                beta, norm_g[ik][spin] = cg_method(
                    scf, ik, cgform, g[ik][spin], g_old[ik][spin], d_old[ik][spin]
                )
                d[ik][spin] = -atoms.K(g[ik][spin], ik) + beta * d_old[ik][spin]
                W[ik][spin] = W[ik][spin] + betat * d[ik][spin]
                gt = grad(scf, ik, spin, W, **scf._precomputed)
                beta = abs(
                    betat
                    * dotprod(g[ik][spin], d[ik][spin])
                    / dotprod(g[ik][spin] - gt, d[ik][spin])
                )
                g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]
                W[ik][spin] = W[ik][spin] + beta * d[ik][spin]

        c = cost(scf, W)
        # If the energy does not go down use the steepest descent step and recalculate the energy
        if c > costs[-1]:
            W = W_old
            for ik in range(atoms.kpts.Nk):
                for spin in range(atoms.occ.Nspin):
                    W[ik][spin] = W[ik][spin] - betat * g[ik][spin]
            c = cost(scf, W)
            costs.append(c)
            # Do not print cg and linmin if we do the sd step
            if condition(scf, "sd", costs, norm_g=norm_g):
                break
        else:
            costs.append(c)
            if condition(scf, "pccg", costs, linmin, cg, norm_g):
                break
    return costs, W


#: Map minimizer names with their respective implementation.
IMPLEMENTED = {
    "sd": sd,
    "lm": lm,
    "pclm": pclm,
    "cg": cg,
    "pccg": pccg,
    "auto": auto,
}
