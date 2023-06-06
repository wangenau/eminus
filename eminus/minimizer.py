#!/usr/bin/env python3
'''Minimization algorithms.'''
import logging

import numpy as np

from .dft import get_grad, get_grad_field, get_n_spin, get_n_total, get_tau, orth, solve_poisson
from .energies import get_E
from .logger import name
from .utils import dotprod
from .xc import get_xc


def scf_step(scf):
    '''Perform one SCF step for a DFT calculation.

    Calculating intermediate results speeds up the energy and gradient calculation.
    This function is similar to H_precompute but will set all variables and energies in the SCF
    class and returns the total energy.

    Args:
        scf: SCF object.

    Returns:
        float: Total energy.
    '''
    atoms = scf.atoms
    scf.Y = orth(atoms, scf.W)
    scf.n_spin = get_n_spin(atoms, scf.Y)
    scf.n = get_n_total(atoms, scf.Y, scf.n_spin)
    if 'gga' in scf.xc_type:
        scf.dn_spin = get_grad_field(atoms, scf.n_spin)
    if scf.xc_type == 'meta-gga':
        scf.tau = get_tau(atoms, scf.Y)
    scf.phi = solve_poisson(atoms, scf.n)
    scf.exc, scf.vxc, scf.vsigma, scf.vtau = get_xc(scf.xc, scf.n_spin, atoms.Nspin, scf.dn_spin,
                                                    scf.tau)
    scf.precomputed = {'dn_spin': scf.dn_spin, 'phi': scf.phi, 'vxc': scf.vxc, 'vsigma': scf.vsigma,
                       'vtau': scf.vtau}
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


def linmin_test(g, d):
    '''Do the line minimization test.

    Calculate the cosine of the angle between g and d.

    Args:
        g (ndarray): Current gradient.
        d (ndarray): Previous search direction.

    Returns:
        float: Linmin angle.
    '''
    # cos = A B / |A| |B|
    return dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))


def cg_test(atoms, g, g_old, precondition=True):
    '''Test the gradient-orthogonality theorem, i.e., g and g_old should be orthogonal.

    Calculate the cosine of the angle between g and g_old. For an angle of 90 deg the cosine goes to
    zero.

    Args:
        atoms: Atoms object.
        g (ndarray): Current gradient.
        g_old (ndarray): Previous gradient.

    Keyword Args:
        precondition (bool): Weather to use a preconditioner.

    Returns:
        float: CG angle.
    '''
    # cos = A B / |A| |B|
    if precondition:
        Kg, Kg_old = atoms.K(g), atoms.K(g_old)
    else:
        Kg, Kg_old = g, g_old
    return dotprod(g, Kg_old) / np.sqrt(dotprod(g, Kg) * dotprod(g_old, Kg_old))


def cg_method(scf, g, g_old, d_old, precondition=True):
    '''Do different variants of the conjugate gradient method.

    Args:
        scf: SCF object.
        g (ndarray): Current gradient.
        g_old (ndarray): Previous gradient.
        d_old (ndarray): Previous search direction.

    Keyword Args:
        precondition (bool): Weather to use a preconditioner.

    Returns:
        float: Conjugate scalar.
    '''
    atoms = scf.atoms

    if precondition:
        Kg, Kg_old = atoms.K(g), atoms.K(g_old)
    else:
        Kg, Kg_old = g, g_old
    if scf.cgform == 1:  # Fletcher-Reeves
        return dotprod(g, Kg) / dotprod(g_old, Kg_old)
    elif scf.cgform == 2:  # Polak-Ribiere
        return dotprod(g - g_old, Kg) / dotprod(g_old, Kg_old)
    elif scf.cgform == 3:  # Hestenes-Stiefel
        return dotprod(g - g_old, Kg) / dotprod(g - g_old, d_old)
    elif scf.cgform == 4:  # Dai-Yuan
        return dotprod(g, Kg) / dotprod(g - g_old, d_old)


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
        betat (float): Step size.

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
            g = grad(scf, spin, scf.W, **scf.precomputed)
            scf.W[spin] = scf.W[spin] - betat * g
    return costs


@name('preconditioned line minimization')
def pclm(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5,
         precondition=True):
    '''Preconditioned line minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        precondition (bool): Weather to use a preconditioner.

    Returns:
        list: Total energies per SCF cycle.
    '''
    atoms = scf.atoms
    costs = []

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    beta = np.empty((atoms.Nspin, 1, 1))
    # Gradients that need to be saved for each spin
    d = np.empty_like(scf.W, dtype=complex)

    for _ in range(Nit):
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, **scf.precomputed)
            # Calculate linmin each spin separately
            if scf.log.level <= logging.DEBUG and Nit > 0:
                linmin[spin] = linmin_test(g, d[spin])
            if precondition:
                d[spin] = -atoms.K(g)
            else:
                d[spin] = -g
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])

        scf.W = scf.W + beta * d
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin):
            break
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
        betat (float): Step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    return pclm(scf, Nit, cost, grad, condition, betat, precondition=False)


@name('preconditioned conjugate-gradient minimization')
def pccg(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5,
         precondition=True):
    '''Preconditioned conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.
        precondition (bool): Weather to use a preconditioner.

    Returns:
        list: Total energies per SCF cycle.
    '''
    atoms = scf.atoms
    costs = []

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    cg = np.empty(atoms.Nspin)
    beta = np.empty((atoms.Nspin, 1, 1))
    # Gradients that need to be saved for each spin
    d = np.empty_like(scf.W, dtype=complex)
    d_old = np.empty_like(scf.W, dtype=complex)
    g_old = np.empty_like(scf.W, dtype=complex)

    # Do the first step without the linmin and cg tests, and without the cg_method
    for spin in range(atoms.Nspin):
        g = grad(scf, spin, scf.W)
        if precondition:
            d[spin] = -atoms.K(g)
        else:
            d[spin] = -g
        gt = grad(scf, spin, scf.W + betat * d[spin])
        beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
        d_old[spin] = d[spin]
        g_old[spin] = g

    # Update wave functions after calculating the gradients for each spin
    scf.W = scf.W + beta * d
    c = cost(scf)
    costs.append(c)
    condition(scf, costs)

    for _ in range(1, Nit):
        for spin in range(atoms.Nspin):
            g = grad(scf, spin, scf.W, **scf.precomputed)
            # Calculate linmin and cg for each spin separately
            if scf.log.level <= logging.DEBUG:
                linmin[spin] = linmin_test(g, d[spin])
                cg[spin] = cg_test(atoms, g, g_old[spin])
            beta[spin] = cg_method(scf, g, g_old[spin], d_old[spin])
            if precondition:
                d[spin] = -atoms.K(g) + beta[spin] * d_old[spin]
            else:
                d[spin] = -g + beta[spin] * d_old[spin]
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g, d[spin]) / dotprod(g - gt, d[spin])
            d_old[spin] = d[spin]
            g_old[spin] = g

        scf.W = scf.W + beta * d
        c = cost(scf)
        costs.append(c)
        if condition(scf, costs, linmin, cg):
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
        betat (float): Step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    return pccg(scf, Nit, cost, grad, condition, betat, precondition=False)


@name('auto minimization')
def auto(scf, Nit, cost=scf_step, grad=get_grad, condition=check_energies, betat=3e-5):
    '''Automatic preconditioned conjugate-gradient minimization algorithm.

    This function chooses an sd step over the pccg step if the energy goes up.

    Args:
        scf: SCF object.
        Nit (int): Maximum number of SCF steps.

    Keyword Args:
        cost (Callable): Function that will run every SCF step.
        grad (Callable): Function that calculates the respective gradient.
        condition (Callable): Function to check and log the convergence condition.
        betat (float): Step size.

    Returns:
        list: Total energies per SCF cycle.
    '''
    atoms = scf.atoms
    costs = []

    # Scalars that need to be saved for each spin
    linmin = np.empty(atoms.Nspin)
    cg = np.empty(atoms.Nspin)
    beta = np.empty((atoms.Nspin, 1, 1))
    # Gradients that need to be saved for each spin
    g = np.empty_like(scf.W, dtype=complex)
    d = np.empty_like(scf.W, dtype=complex)
    d_old = np.empty_like(scf.W, dtype=complex)
    g_old = np.empty_like(scf.W, dtype=complex)

    # Start with a cost calculation, also print the minimization type
    c = cost(scf)
    costs.append(c)
    print('Type\t', end='')
    condition(scf, costs)

    # Do the first step without the linmin and cg tests, and without the cg_method
    for spin in range(atoms.Nspin):
        g[spin] = grad(scf, spin, scf.W, **scf.precomputed)
        d[spin] = -atoms.K(g[spin])
        gt = grad(scf, spin, scf.W + betat * d[spin])
        beta[spin] = betat * dotprod(g[spin], d[spin]) / dotprod(g[spin] - gt, d[spin])
        d_old[spin] = d[spin]
        g_old[spin] = g[spin]

    # Update wave functions after calculating the gradients for each spin, save the wave function
    W_old = np.copy(scf.W)
    scf.W = scf.W + beta * d
    c = cost(scf)
    # If the energy does not go down, use the steepest descent step and recalculate the energy
    if c > costs[-1]:
        scf.W = W_old
        for spin in range(atoms.Nspin):
            scf.W[spin] = scf.W[spin] - betat * g[spin]
        c = cost(scf)
        print('sd\t', end='')
    else:
        print('pccg\t', end='')
    costs.append(c)
    if condition(scf, costs):
        return costs

    for _ in range(2, Nit):
        for spin in range(atoms.Nspin):
            g[spin] = grad(scf, spin, scf.W, **scf.precomputed)
            # Calculate linmin and cg for each spin separately
            if scf.log.level <= logging.DEBUG:
                linmin[spin] = linmin_test(g, d[spin])
                cg[spin] = cg_test(atoms, g, g_old[spin])
            beta[spin] = cg_method(scf, g, g_old[spin], d_old[spin])
            d[spin] = -atoms.K(g[spin]) + beta[spin] * d_old[spin]
            gt = grad(scf, spin, scf.W + betat * d[spin])
            beta[spin] = betat * dotprod(g[spin], d[spin]) / dotprod(g[spin] - gt, d[spin])
            d_old[spin] = d[spin]
            g_old[spin] = g[spin]

        W_old = np.copy(scf.W)
        scf.W = scf.W + beta * d
        c = cost(scf)
        # If the energy does not go down use the steepest descent step and recalculate the energy
        if c > costs[-1]:
            scf.W = W_old
            for spin in range(atoms.Nspin):
                scf.W[spin] = scf.W[spin] - betat * g[spin]
            c = cost(scf)
            print('sd\t', end='')
            costs.append(c)
            # Do not print cg and linmin if we chose the sd step
            if condition(scf, costs):
                break
        else:
            print('pccg\t', end='')
            costs.append(c)
            if condition(scf, costs, linmin, cg):
                break
    return costs


IMPLEMENTED = {
    'sd': sd,
    'lm': lm,
    'pclm': pclm,
    'cg': cg,
    'pccg': pccg,
    'auto': auto
}
