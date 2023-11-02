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


def scf_step(scf):
    """Perform one SCF step for a band minimization calculation.

    Args:
        scf: SCF object.

    Returns:
        float: Total energy.
    """
    atoms = scf.atoms
    scf.Y = orth(atoms, scf.W)
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
    W = orth(atoms, scf.W)
    HW = H(scf, ik, spin, W, **kwargs)
    WHW = W[ik][spin].conj().T @ HW
    OW = atoms.O(W[ik][spin])
    U = W[ik][spin].conj().T @ OW
    invU = inv(U)
    U12 = sqrtm(invU)
    return atoms.wk[ik] * ((HW - OW @ WHW) @ U12)


@name('steepest descent minimization')
def sd(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, **kwargs):
    """Steepest descent minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        **kwargs: Throwaway arguments.

    Returns:
        list: Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    for _ in range(Nit):
        c = cost(scf)
        costs.append(c)
        if condition(scf, 'sd', costs):
            break
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, scf.W, **scf._precomputed)
                scf.W[ik][spin] = scf.W[ik][spin] - betat * g
    return costs


@name('preconditioned line minimization')
def pclm(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5,
         precondition=True, **kwargs):
    """Preconditioned line minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        precondition (bool): Weather to use a preconditioner.
        **kwargs: Throwaway arguments.

    Returns:
        list: Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    linmin = None
    d = np.empty_like(scf.W[0][0], dtype=complex)

    if precondition:
        method = 'pclm'
    else:
        method = 'lm'

    for _ in range(Nit):
        c = cost(scf)
        costs.append(c)
        if condition(scf, method, costs, linmin):
            break
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, scf.W, **scf._precomputed)
                if scf.log.level <= logging.DEBUG and Nit > 0:
                    linmin = linmin_test(g, d)
                if precondition:
                    d = -atoms.K(g, ik)
                else:
                    d = -g
                scf.W[ik][spin] = scf.W[ik][spin] + betat * d
                gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
                beta = betat * dotprod(g, d) / dotprod(g - gt, d)
                scf.W[ik][spin] = scf.W[ik][spin] + beta * d
    return costs


@name('line minimization')
def lm(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, **kwargs):
    """Line minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        **kwargs: Throwaway arguments.

    Returns:
        list: Total energies per SCF cycle.
    """
    return pclm(scf, Nit, cost, grad, condition, betat, precondition=False)


@name('preconditioned conjugate-gradient minimization')
def pccg(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, cgform=1,
         precondition=True):
    """Preconditioned conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        cgform (int): Conjugate gradient form.
        precondition (bool): Weather to use a preconditioner.

    Returns:
        list: Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    linmin = None
    cg = None
    norm_g = None
    d_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    g_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]

    if precondition:
        method = 'pccg'
    else:
        method = 'cg'

    c = cost(scf)
    costs.append(c)
    condition(scf, method, costs)
    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(len(atoms.wk)):
        for spin in range(atoms.occ.Nspin):
            g = grad(scf, ik, spin, scf.W, **scf._precomputed)
            if precondition:
                d = -atoms.K(g, ik)
            else:
                d = -g
            scf.W[ik][spin] = scf.W[ik][spin] + betat * d
            gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
            beta = betat * dotprod(g, d) / dotprod(g - gt, d)
            g_old[ik][spin], d_old[ik][spin] = g, d
            scf.W[ik][spin] = scf.W[ik][spin] + beta * d

    for _ in range(1, Nit):
        c = cost(scf)
        costs.append(c)
        if condition(scf, method, costs, linmin, cg, norm_g):
            break
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf.log.level <= logging.DEBUG:
                    linmin = linmin_test(g, d)
                    cg = cg_test(atoms, g, g_old[ik][spin], precondition)
                beta, norm_g = cg_method(scf, ik, cgform, g, g_old[ik][spin], d_old[ik][spin], precondition)
                if precondition:
                    d = -atoms.K(g, ik) + beta * d_old[ik][spin]
                else:
                    d = -g + beta * d_old[ik][spin]
                scf.W[ik][spin] = scf.W[ik][spin] + betat * d
                gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
                beta = betat * dotprod(g, d) / dotprod(g - gt, d)
                g_old[ik][spin], d_old[ik][spin] = g, d
                scf.W[ik][spin] = scf.W[ik][spin] + beta * d
    return costs


@name('conjugate-gradient minimization')
def cg(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, cgform=1):
    """Conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        cgform (int): Conjugate gradient form.

    Returns:
        list: Total energies per SCF cycle.
    """
    return pccg(scf, Nit, cost, grad, condition, betat, cgform, precondition=False)


@name('auto minimization')
def auto(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, cgform=1):
    """Automatic preconditioned conjugate-gradient minimization algorithm.

    This function chooses an sd step over the pccg step if the energy goes up.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        cgform (int): Conjugate gradient form.

    Returns:
        list: Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    linmin = None
    cg = None
    norm_g = None
    g = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    d_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    g_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]

    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(len(atoms.wk)):
        for spin in range(atoms.occ.Nspin):
            g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
            d = -atoms.K(g[ik][spin], ik)
            scf.W[ik][spin] = scf.W[ik][spin] + betat * d
            gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
            beta = betat * dotprod(g[ik][spin], d) / dotprod(g[ik][spin] - gt, d)
            g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d
            scf.W[ik][spin] = scf.W[ik][spin] + beta * d

    c = cost(scf)
    costs.append(c)
    if condition(scf, 'pccg', costs):
        return costs

    for _ in range(1, Nit):
        W_old = copy.deepcopy(scf.W)
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf.log.level <= logging.DEBUG:
                    linmin = linmin_test(g[ik][spin], d)
                    cg = cg_test(atoms, g[ik][spin], g_old[ik][spin])
                beta, norm_g = cg_method(scf, ik, cgform, g[ik][spin], g_old[ik][spin], d_old[ik][spin])
                d = -atoms.K(g[ik][spin], ik) + beta * d_old[ik][spin]
                scf.W[ik][spin] = scf.W[ik][spin] + betat * d
                gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
                beta = betat * dotprod(g[ik][spin], d) / dotprod(g[ik][spin] - gt, d)
                g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d
                scf.W[ik][spin] = scf.W[ik][spin] + beta * d

        c = cost(scf)
        # If the energy does not go down use the steepest descent step and recalculate the energy
        if c > costs[-1]:
            scf.W = W_old
            for ik in range(len(atoms.wk)):
                for spin in range(atoms.occ.Nspin):
                    scf.W[ik][spin] = scf.W[ik][spin] - betat * g[ik][spin]
            c = cost(scf)
            costs.append(c)
            # Do not print cg and linmin if we do the sd step
            if condition(scf, 'sd', costs, norm_g=norm_g):
                break
        else:
            costs.append(c)
            if condition(scf, 'pccg', costs, linmin, cg, norm_g):
                break
    return costs


#: Map minimizer names with their respective implementation.
IMPLEMENTED = {
    'sd': sd,
    'lm': lm,
    'pclm': pclm,
    'cg': cg,
    'pccg': pccg,
    'auto': auto
}
