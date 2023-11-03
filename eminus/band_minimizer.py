#!/usr/bin/env python3
"""Minimization algorithms."""
import copy
import logging

import numpy as np
from scipy.linalg import inv, sqrtm

from .dft import H, orth
from .energies import get_Eband
from .logger import name
from .minimizer import cg_method, cg_test, check_convergence, linmin_test
from .utils import dotprod


def scf_step(scf, W):
    """Perform one SCF step for a band minimization calculation.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        float: Total energy.
    """
    atoms = scf.atoms
    scf.Y = orth(atoms, W)
    return get_Eband(scf, scf.Y, **scf._precomputed)


def get_grad(scf, ik, spin, W, **kwargs):
    """Calculate the band energy gradient with respect to W.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        ik (int): k-point index.
        spin (int): Spin variable to track weather to do the calculation for spin up or down.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`H`.

    Returns:
        ndarray: Gradient.
    """
    atoms = scf.atoms
    W = orth(atoms, W)
    HW = H(scf, ik, spin, W, **kwargs)
    WHW = W[ik][spin].conj().T @ HW
    OW = atoms.O(W[ik][spin])
    U = W[ik][spin].conj().T @ OW
    invU = inv(U)
    U12 = sqrtm(invU)
    return atoms.kpts.wk[ik] * ((HW - OW @ WHW) @ U12)


@name('steepest descent minimization')
def sd(scf, W, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=1, **kwargs):
    """Steepest descent minimization algorithm.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        **kwargs: Throwaway arguments.

    Returns:
        tuple[list, ndarray]: Energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    for _ in range(Nit):
        c = cost(scf, W)
        costs.append(c)
        if condition(scf, 'sd', costs):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, W, **scf._precomputed)
                W[ik][spin] = W[ik][spin] - betat * g
    return costs, W


@name('preconditioned line minimization')
def pclm(scf, W, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=1,
         precondition=True, **kwargs):
    """Preconditioned line minimization algorithm.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        precondition (bool): Weather to use a preconditioner.
        **kwargs: Throwaway arguments.

    Returns:
        tuple[list, ndarray]: Energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    linmin = None
    d = np.empty_like(W[0][0], dtype=complex)

    if precondition:
        method = 'pclm'
    else:
        method = 'lm'

    for _ in range(Nit):
        c = cost(scf, W)
        costs.append(c)
        if condition(scf, method, costs, linmin):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, W, **scf._precomputed)
                if scf.log.level <= logging.DEBUG and Nit > 0:
                    linmin = linmin_test(g, d)
                if precondition:
                    d = -atoms.K(g, ik)
                else:
                    d = -g
                W[ik][spin] = W[ik][spin] + betat * d
                gt = grad(scf, ik, spin, W, **scf._precomputed)
                beta = betat * dotprod(g, d) / dotprod(g - gt, d)
                W[ik][spin] = W[ik][spin] + beta * d
    return costs, W


@name('line minimization')
def lm(scf, W, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=1, **kwargs):
    """Line minimization algorithm.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        **kwargs: Throwaway arguments.

    Returns:
        tuple[list, ndarray]: Energies per SCF cycle and optimized expansion coefficients.
    """
    return pclm(scf, W, Nit, cost, grad, condition, betat, precondition=False)


@name('preconditioned conjugate-gradient minimization')
def pccg(scf, W, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=1, cgform=1,
         precondition=True):
    """Preconditioned conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        cgform (int): Conjugate gradient form.
        precondition (bool): Weather to use a preconditioner.

    Returns:
        tuple[list, ndarray]: Energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    linmin = None
    cg = None
    norm_g = None
    d_old = [np.empty_like(W[ik], dtype=complex) for ik in range(atoms.kpts.Nk)]
    g_old = [np.empty_like(W[ik], dtype=complex) for ik in range(atoms.kpts.Nk)]

    if precondition:
        method = 'pccg'
    else:
        method = 'cg'

    c = cost(scf, W)
    costs.append(c)
    condition(scf, method, costs)
    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            g = grad(scf, ik, spin, W, **scf._precomputed)
            if precondition:
                d = -atoms.K(g, ik)
            else:
                d = -g
            W[ik][spin] = W[ik][spin] + betat * d
            gt = grad(scf, ik, spin, W, **scf._precomputed)
            beta = betat * dotprod(g, d) / dotprod(g - gt, d)
            g_old[ik][spin], d_old[ik][spin] = g, d
            W[ik][spin] = W[ik][spin] + beta * d

    for _ in range(1, Nit):
        c = cost(scf, W)
        costs.append(c)
        if condition(scf, method, costs, linmin, cg, norm_g):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf.log.level <= logging.DEBUG:
                    linmin = linmin_test(g, d)
                    cg = cg_test(atoms, g, g_old[ik][spin], precondition)
                beta, norm_g = cg_method(scf, ik, cgform, g, g_old[ik][spin], d_old[ik][spin],
                                         precondition)
                if precondition:
                    d = -atoms.K(g, ik) + beta * d_old[ik][spin]
                else:
                    d = -g + beta * d_old[ik][spin]
                W[ik][spin] = W[ik][spin] + betat * d
                gt = grad(scf, ik, spin, W, **scf._precomputed)
                beta = betat * dotprod(g, d) / dotprod(g - gt, d)
                g_old[ik][spin], d_old[ik][spin] = g, d
                W[ik][spin] = W[ik][spin] + beta * d
    return costs, W


@name('conjugate-gradient minimization')
def cg(scf, W, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=1, cgform=1):
    """Conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        cgform (int): Conjugate gradient form.

    Returns:
        tuple[list, ndarray]: Energies per SCF cycle and optimized expansion coefficients.
    """
    return pccg(scf, W, Nit, cost, grad, condition, betat, cgform, precondition=False)


@name('auto minimization')
def auto(scf, W, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=1, cgform=1):
    """Automatic preconditioned conjugate-gradient minimization algorithm.

    This function chooses an sd step over the pccg step if the energy goes up.

    Args:
        scf: SCF object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        cgform (int): Conjugate gradient form.

    Returns:
        tuple[list, ndarray]: Energies per SCF cycle and optimized expansion coefficients.
    """
    atoms = scf.atoms
    costs = []

    linmin = None
    cg = None
    norm_g = None
    g = [np.empty_like(W[ik], dtype=complex) for ik in range(atoms.kpts.Nk)]
    d_old = [np.empty_like(W[ik], dtype=complex) for ik in range(atoms.kpts.Nk)]
    g_old = [np.empty_like(W[ik], dtype=complex) for ik in range(atoms.kpts.Nk)]

    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            g[ik][spin] = grad(scf, ik, spin, W, **scf._precomputed)
            d = -atoms.K(g[ik][spin], ik)
            W[ik][spin] = W[ik][spin] + betat * d
            gt = grad(scf, ik, spin, W, **scf._precomputed)
            beta = betat * dotprod(g[ik][spin], d) / dotprod(g[ik][spin] - gt, d)
            g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d
            W[ik][spin] = W[ik][spin] + beta * d

    c = cost(scf, W)
    costs.append(c)
    if condition(scf, 'pccg', costs):
        return costs

    for _ in range(1, Nit):
        W_old = copy.deepcopy(W)
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf.log.level <= logging.DEBUG:
                    linmin = linmin_test(g[ik][spin], d)
                    cg = cg_test(atoms, g[ik][spin], g_old[ik][spin])
                beta, norm_g = cg_method(scf, ik, cgform, g[ik][spin], g_old[ik][spin],
                                         d_old[ik][spin])
                d = -atoms.K(g[ik][spin], ik) + beta * d_old[ik][spin]
                W[ik][spin] = W[ik][spin] + betat * d
                gt = grad(scf, ik, spin, W, **scf._precomputed)
                beta = betat * dotprod(g[ik][spin], d) / dotprod(g[ik][spin] - gt, d)
                g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d
                W[ik][spin] = W[ik][spin] + beta * d

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
            if condition(scf, 'sd', costs, norm_g=norm_g):
                break
        else:
            costs.append(c)
            if condition(scf, 'pccg', costs, linmin, cg, norm_g):
                break
    return costs, W


#: Map minimizer names with their respective implementation.
IMPLEMENTED = {
    'sd': sd,
    'lm': lm,
    'pclm': pclm,
    'cg': cg,
    'pccg': pccg,
    'auto': auto
}
