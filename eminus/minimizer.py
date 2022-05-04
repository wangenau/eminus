#!/usr/bin/env python3
'''Minimization algorithms.'''
import logging

import numpy as np

from .dft import get_grad, get_n_total, orth, solve_poisson
from .energies import get_E
from .logger import name
from .utils import dotprod
from .xc import get_xc


def scf_step(scf):
    '''Perform one SCF step for a DFT calculation.

    Calculating intermediate results speeds up the energy and gradient calculation.

    Args:
        scf: SCF object.

    Returns:
        float: Total energy.
    '''
    scf.Y = orth(scf.atoms, scf.W)
    scf.n = get_n_total(scf.atoms, scf.Y)
    scf.phi = solve_poisson(scf.atoms, scf.n)
    scf.exc, scf.vxc = get_xc(scf.xc, scf.n)
    return get_E(scf)


def check_energies(scf, Elist, linmin='', cg=''):
    '''Check the energies for every SCF cycle and handle the output.

    Args:
        scf: SCF object.
        Elist (list): Total energies per SCF step.

    Keyword Args:
        linmin (float): Cosine between previous search direction and current gradient.
        cg (float): Conjugate-gradient orthogonality.

    Returns:
        bool: Convergence condition.
    '''
    iteration = len(Elist)

    # Output handling
    if not isinstance(linmin, str):
        linmin = f' \tlinmin-test: {linmin:+.7f}'
    if not isinstance(cg, str):
        cg = f' \tcg-test: {cg:+.7f}'
    if scf.log.level <= logging.DEBUG:
        scf.log.debug(f'Iteration: {iteration}  \tEtot: {scf.energies.Etot:+.7f}{linmin}{cg}')
    else:
        scf.log.info(f'Iteration: {iteration}  \tEtot: {scf.energies.Etot:+.7f}')

    # Check for convergence
    if iteration > 1:
        if abs(Elist[-2] - Elist[-1]) < scf.etol:
            return True
        elif Elist[-1] > Elist[-2]:
            scf.log.warning('Total energy is not decreasing.')
    return False


@name('steepest descent minimization')
def sd(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5):
    '''Steepest descent minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): SCF step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    costs = []

    for _ in range(Nit):
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs):
            break
        g = grad(scf, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
        scf.W = scf.W - betat * g
    return costs


@name('line minimization')
def lm(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5):
    '''Line minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): SCF step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    costs = []

    # Do the first step without the linmin test
    g = grad(scf, scf.W)
    d = -g
    gt = grad(scf, scf.W + betat * d)
    beta = betat * dotprod(g, d) / dotprod(g - gt, d)

    scf.W = scf.W + beta * d
    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        g = grad(scf, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -g
        gt = grad(scf, scf.W + betat * d)
        beta = betat * dotprod(g, d) / dotprod(g - gt, d)

        scf.W = scf.W + beta * d
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin):
            break
    return costs


@name('preconditioned line minimization')
def pclm(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5):
    '''Preconditioned line minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): SCF step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    atoms = scf.atoms
    costs = []

    # Do the first step without the linmin test
    g = grad(scf, scf.W)
    d = -atoms.K(g)
    gt = grad(scf, scf.W + betat * d)
    beta = betat * dotprod(g, d) / dotprod(g - gt, d)

    scf.W = scf.W + beta * d
    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        g = grad(scf, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
        linmin = dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))
        d = -atoms.K(g)
        gt = grad(scf, scf.W + betat * d)
        beta = betat * dotprod(g, d) / dotprod(g - gt, d)

        scf.W = scf.W + beta * d
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin):
            break
    return costs


@name('conjugate-gradient minimization')
def cg(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5):
    '''Conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): SCF step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    costs = []

    # Do the first step without the linmin and cg test
    g = grad(scf, scf.W)
    d = -g
    gt = grad(scf, scf.W + betat * d)
    beta = betat * dotprod(g, d) / dotprod(g - gt, d)
    d_old = d
    g_old = g

    scf.W = scf.W + beta * d
    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        g = grad(scf, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
        linmin = dotprod(g, d_old) / np.sqrt(dotprod(g, g) * dotprod(d_old, d_old))
        cg = dotprod(g, g_old) / np.sqrt(dotprod(g, g) *
             dotprod(g_old, g_old))
        if scf.cgform == 1:  # Fletcher-Reeves
            beta = dotprod(g, g) / dotprod(g_old, g_old)
        elif scf.cgform == 2:  # Polak-Ribiere
            beta = dotprod(g - g_old, g) / dotprod(g_old, g_old)
        elif scf.cgform == 3:  # Hestenes-Stiefel
            beta = dotprod(g - g_old, g) / dotprod(g - g_old, d_old)
        d = -g + beta * d_old
        gt = grad(scf, scf.W + betat * d)
        beta = betat * dotprod(g, d) / dotprod(g - gt, d)
        d_old = d
        g_old = g

        scf.W = scf.W + beta * d
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin, cg):
            break
    return costs


@name('preconditioned conjugate-gradient minimization')
def pccg(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5):
    '''Preconditioned conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): SCF step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    atoms = scf.atoms
    costs = []

    # Do the first step without the linmin and cg test
    g = grad(scf, scf.W)
    d = -atoms.K(g)
    gt = grad(scf, scf.W + betat * d)
    beta = betat * dotprod(g, d) / dotprod(g - gt, d)
    d_old = d
    g_old = g

    scf.W = scf.W + beta * d
    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        g = grad(scf, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
        linmin = dotprod(g, d_old) / np.sqrt(dotprod(g, g) * dotprod(d_old, d_old))
        cg = dotprod(g, atoms.K(g_old)) / np.sqrt(dotprod(g, atoms.K(g)) *
             dotprod(g_old, atoms.K(g_old)))
        if scf.cgform == 1:  # Fletcher-Reeves
            beta = dotprod(g, atoms.K(g)) / dotprod(g_old, atoms.K(g_old))
        elif scf.cgform == 2:  # Polak-Ribiere
            beta = dotprod(g - g_old, atoms.K(g)) / dotprod(g_old, atoms.K(g_old))
        elif scf.cgform == 3:  # Hestenes-Stiefel
            beta = dotprod(g - g_old, atoms.K(g)) / dotprod(g - g_old, d_old)
        d = -atoms.K(g) + beta * d_old
        gt = grad(scf, scf.W + betat * d)
        beta = betat * dotprod(g, d) / dotprod(g - gt, d)
        d_old = d
        g_old = g

        scf.W = scf.W + beta * d
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin, cg):
            break
    return costs
