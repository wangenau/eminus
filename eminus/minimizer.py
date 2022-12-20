#!/usr/bin/env python3
'''Minimization algorithms.'''
import logging

import numpy as np

from .dft import get_grad, get_n_spin, get_n_total, orth, solve_poisson
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
    atoms = scf.atoms
    scf.Y = orth(atoms, scf.W)
    scf.n = get_n_total(atoms, scf.Y)
    scf.n_spin = get_n_spin(atoms, scf.Y, scf.n)
    scf.phi = solve_poisson(atoms, scf.n)
    scf.exc, scf.vxc = get_xc(scf.xc, scf.n_spin, atoms.Nspin)
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

    if scf.log.level <= logging.DEBUG:
        with np.printoptions(formatter={'float': '{:+0.3e}'.format}):
            # Output handling
            if not isinstance(linmin, str):
                linmin = f' \tlinmin-test: {linmin}'
            if not isinstance(cg, str):
                cg = f' \tcg-test: {cg}'
            scf.log.debug(f'Iteration: {iteration} \tEtot: '
                          f'{scf.energies.Etot:+.{scf.print_precision}f}{linmin}{cg}')
    else:
        scf.log.info(f'Iteration: {iteration} \tEtot: {scf.energies.Etot:+.{scf.print_precision}f}')

    if iteration > 1:
        # Check for convergence
        if abs(Elist[-2] - Elist[-1]) < scf.etol:
            return True
        # Check if the current energy is lower than any of the last three values
        if np.any(np.asarray(Elist[-4:-1]) < Elist[-1]):
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
    atoms = scf.atoms
    costs = []

    for _ in range(Nit):
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs):
            break
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
            scf.W[spin] = scf.W[spin] - betat * g
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
    atoms = scf.atoms
    costs = []

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    beta = np.empty(atoms.Nspin)

    # Gradients that need to be saved for each spin
    d = np.empty_like(scf.W, dtype=complex)

    # Do the first step without the linmin test
    for spin in range(atoms.Nspin):
        g = grad(scf, spin, scf.W)
        d[spin] = -g
        gt = grad(scf, spin, scf.W + betat * d[spin])
        beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
    # Update wave functions after calculating the gradients for each spin
    for spin in range(atoms.Nspin):
        scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
            # Calculate linmin each spin seperately
            if scf.log.level <= logging.DEBUG:
                linmin[spin] = dotprod(g, d[spin]) / \
                    np.sqrt(dotprod(g, g) * dotprod(d[spin], d[spin]))
            d[spin] = -g
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
        for spin in range(atoms.Nspin):
            scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

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

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    beta = np.empty(atoms.Nspin)

    # Gradients that need to be saved for each spin
    d = np.empty_like(scf.W, dtype=complex)

    # Do the first step without the linmin test
    for spin in range(atoms.Nspin):
        g = grad(scf, spin, scf.W)
        d[spin] = -atoms.K(g)
        gt = grad(scf, spin, scf.W + betat * d[spin])
        beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
    # Update wave functions after calculating the gradients for each spin
    for spin in range(atoms.Nspin):
        scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
            # Calculate linmin each spin seperately
            if scf.log.level <= logging.DEBUG:
                linmin[spin] = dotprod(g, d[spin]) / \
                    np.sqrt(dotprod(g, g) * dotprod(d[spin], d[spin]))
            d[spin] = -atoms.K(g)
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
        for spin in range(atoms.Nspin):
            scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

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
    atoms = scf.atoms
    costs = []

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    cg = np.empty(atoms.Nspin)
    beta = np.empty(atoms.Nspin)

    # Gradients that need to be saved for each spin
    d = np.empty_like(scf.W, dtype=complex)
    d_old = np.empty_like(scf.W, dtype=complex)
    g_old = np.empty_like(scf.W, dtype=complex)

    # Do the first step without the linmin and cg test
    for spin in range(atoms.Nspin):
        g = grad(scf, spin, scf.W)
        d[spin] = -g
        gt = grad(scf, spin, scf.W + betat * d[spin])
        beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
        d_old[spin] = d[spin]
        g_old[spin] = g
    # Update wave functions after calculating the gradients for each spin
    for spin in range(atoms.Nspin):
        scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
            # Calculate linmin and cg for each spin seperately
            if scf.log.level <= logging.DEBUG:
                linmin[spin] = dotprod(g, d_old[spin]) / \
                    np.sqrt(dotprod(g, g) * dotprod(d_old[spin], d_old[spin]))
                cg[spin] = dotprod(g, g_old[spin]) / \
                    np.sqrt(dotprod(g, g) * dotprod(g_old[spin], g_old[spin]))
            if scf.cgform == 1:  # Fletcher-Reeves
                beta[spin] = dotprod(g, g) / dotprod(g_old[spin], g_old[spin])
            elif scf.cgform == 2:  # Polak-Ribiere
                beta[spin] = dotprod(g - g_old[spin], g) / \
                    dotprod(g_old[spin], g_old[spin])
            elif scf.cgform == 3:  # Hestenes-Stiefel
                beta[spin] = dotprod(g - g_old[spin], g) / \
                    dotprod(g - g_old[spin], d_old[spin])
            d[spin] = -g + beta[spin] * d_old[spin]
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
            d_old[spin] = d[spin]
            g_old[spin] = g
        for spin in range(atoms.Nspin):
            scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

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

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    cg = np.empty(atoms.Nspin)
    beta = np.empty(atoms.Nspin)

    # Gradients that need to be saved for each spin
    d = np.empty_like(scf.W, dtype=complex)
    d_old = np.empty_like(scf.W, dtype=complex)
    g_old = np.empty_like(scf.W, dtype=complex)

    # Do the first step without the linmin and cg test
    for spin in range(atoms.Nspin):
        g = grad(scf, spin, scf.W)
        d[spin] = -atoms.K(g)
        gt = grad(scf, spin, scf.W + betat * d[spin])
        beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
        d_old[spin] = d[spin]
        g_old[spin] = g
    # Update wave functions after calculating the gradients for each spin
    for spin in range(atoms.Nspin):
        scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, scf.Y, scf.n, scf.phi, scf.vxc)
            # Calculate linmin and cg for each spin seperately
            if scf.log.level <= logging.DEBUG:
                linmin[spin] = dotprod(g, d_old[spin]) / \
                    np.sqrt(dotprod(g, g) * dotprod(d_old[spin], d_old[spin]))
                cg[spin] = dotprod(g, atoms.K(g_old[spin])) / \
                    np.sqrt(dotprod(g, atoms.K(g)) * dotprod(g_old[spin], atoms.K(g_old[spin])))
            if scf.cgform == 1:  # Fletcher-Reeves
                beta[spin] = dotprod(g, atoms.K(g)) / dotprod(g_old[spin], atoms.K(g_old[spin]))
            elif scf.cgform == 2:  # Polak-Ribiere
                beta[spin] = dotprod(g - g_old[spin], atoms.K(g)) / \
                    dotprod(g_old[spin], atoms.K(g_old[spin]))
            elif scf.cgform == 3:  # Hestenes-Stiefel
                beta[spin] = dotprod(g - g_old[spin], atoms.K(g)) / \
                    dotprod(g - g_old[spin], d_old[spin])
            d[spin] = -atoms.K(g) + beta[spin] * d_old[spin]
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
            d_old[spin] = d[spin]
            g_old[spin] = g
        for spin in range(atoms.Nspin):
            scf.W[spin] = scf.W[spin] + beta[spin] * d[spin]

        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin, cg):
            break
    return costs


IMPLEMENTED = {
    'cg': cg,
    'lm': lm,
    'pccg': pccg,
    'pclm': pclm,
    'sd': sd
}
