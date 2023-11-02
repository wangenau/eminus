#!/usr/bin/env python3
"""Minimization algorithms."""
import copy
import logging

import numpy as np

from .dft import get_grad, get_n_spin, get_n_total, orth, solve_poisson
from .energies import get_E
from .gga import get_grad_field, get_tau
from .logger import name
from .utils import dotprod
from .xc import get_xc


def scf_step(scf):
    """Perform one SCF step for a DFT calculation.

    Calculating intermediate results speeds up the energy and gradient calculation.
    This function is similar to H_precompute but will set all variables and energies in the SCF
    class and returns the total energy.

    Args:
        scf: SCF object.

    Returns:
        float: Total energy.
    """
    atoms = scf.atoms
    scf.Y = orth(atoms, scf.W)
    scf.n_spin = get_n_spin(atoms, scf.Y)
    scf.n = get_n_total(atoms, scf.Y, scf.n_spin)
    if 'gga' in scf.xc_type:
        scf.dn_spin = get_grad_field(atoms, scf.n_spin)
    if scf.xc_type == 'meta-gga':
        scf.tau = get_tau(atoms, scf.Y)
    scf.phi = solve_poisson(atoms, scf.n)
    scf.exc, scf.vxc, scf.vsigma, scf.vtau = get_xc(scf.xc, scf.n_spin, atoms.occ.Nspin,
                                                    scf.dn_spin, scf.tau)
    scf._precomputed = {'dn_spin': scf.dn_spin, 'phi': scf.phi, 'vxc': scf.vxc,
                        'vsigma': scf.vsigma, 'vtau': scf.vtau}
    return get_E(scf)


def check_convergence(scf, method, Elist, linmin=None, cg=None, norm_g=None):
    """Check the energies for every SCF cycle and handle the output.

    Args:
        scf: SCF object.
        method (string): Minimization method.
        Elist (list): Total energies per SCF step.

    Keyword Args:
        linmin (ndarray): Cosine between previous search direction and current gradient.
        cg (ndarray): Conjugate-gradient orthogonality.
        norm_g (ndarray): Gradient norm.

    Returns:
        bool: Convergence condition.
    """
    iteration = len(Elist)

    # Print all the data
    print_scf_step(scf, method, Elist, linmin, cg, norm_g)

    if iteration > 1:
        # Check for convergence
        if scf.gradtol is None or norm_g is None:
            if abs(Elist[-2] - Elist[-1]) < scf.etol:
                scf.is_converged = True
                return True
        # If a gradient tolerance has been set we also check norm_g for convergence
        elif abs(Elist[-2] - Elist[-1]) < scf.etol and (norm_g < scf.gradtol).all():
            scf.is_converged = True
            return True
        # Check if the current energy is higher than the last two values
        if (np.asarray(Elist[-3:-1]) < Elist[-1]).all():
            scf.log.warning('Total energy is not decreasing.')
    return False


def print_scf_step(scf, method, Elist, linmin, cg, norm_g):
    """Print the data of one SCF step and the header at the beginning.

    Args:
        scf: SCF object.
        method (string): Minimization method.
        Elist (list): Total energies per SCF step.
        linmin (ndarray): Cosine between previous search direction and current gradient.
        cg (ndarray): Conjugate-gradient orthogonality.
        norm_g (ndarray): Gradient norm.
    """
    iteration = len(Elist)

    # Print a column header at the beginning
    # The ljust values just have been chosen such that the output looks decent
    if iteration == 1:
        header = 'Method'.ljust(8)
        header += 'Iteration'.ljust(11)
        header += 'Etot [Eh]'.ljust(13)
        header += 'dEtot [Eh]'.ljust(13)
        # Print the gradient norm for cg methods
        if method not in ('sd', 'lm', 'pclm'):
            header += '|Gradient|'.ljust(10 * scf.atoms.occ.Nspin + 3)
        # Print extra debugging information if available
        if scf.log.level <= logging.DEBUG:
            if method != 'sd':
                header += 'linmin-test'.ljust(10 * scf.atoms.occ.Nspin + 3)
            if method not in ('sd', 'lm', 'pclm'):
                header += 'cg-test'.ljust(10 * scf.atoms.occ.Nspin + 3)
            scf.log.debug(header)
        else:
            scf.log.info(header)

    # Print the information for every cycle
    # Context manager for printing norm_g, linmin, and cg
    with np.printoptions(formatter={'float': '{:+0.2e}'.format}):
        info = f'{method:<8}{iteration:>8}   {scf.energies.Etot:<+13,.6f}'
        # In the first step we do not have all information yet
        if iteration > 1:
            info += f'{Elist[-2] - Elist[-1]:<+13,.4e}'
            if norm_g is not None:
                info += str(np.ravel(norm_g)).ljust(10 * scf.atoms.occ.Nspin + 3)
            if scf.log.level <= logging.DEBUG:
                if method != 'sd':
                    info += str(linmin).ljust(10 * scf.atoms.occ.Nspin + 3)
                if method not in ('sd', 'lm', 'pclm'):
                    info += str(cg).ljust(10 * scf.atoms.occ.Nspin + 3)
    if scf.log.level <= logging.DEBUG:
        scf.log.debug(info)
    else:
        scf.log.info(info)


def linmin_test(g, d):
    """Do the line minimization test.

    Calculate the cosine of the angle between g and d.

    Reference: https://trond.hjorteland.com/thesis/node26.html

    Args:
        g (ndarray): Current gradient.
        d (ndarray): Previous search direction.

    Returns:
        float: Linmin angle.
    """
    # cos = A B / |A| |B|
    return dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))


def cg_test(atoms, g, g_old, precondition=True):
    """Test the gradient-orthogonality theorem, i.e., g and g_old should be orthogonal.

    Calculate the cosine of the angle between g and g_old. For an angle of 90 degree the cosine goes
    to zero.

    Reference: https://math.uci.edu/~chenlong/CAMtips/CG.html

    Args:
        atoms: Atoms object.
        g (ndarray): Current gradient.
        g_old (ndarray): Previous gradient.

    Keyword Args:
        precondition (bool): Weather to use a preconditioner.

    Returns:
        float: CG angle.
    """
    if precondition:
        Kg, Kg_old = atoms.K(g), atoms.K(g_old)
    else:
        Kg, Kg_old = g, g_old
    # cos = A B / |A| |B|
    return dotprod(g, Kg_old) / np.sqrt(dotprod(g, Kg) * dotprod(g_old, Kg_old))


def cg_method(scf, ik, cgform, g, g_old, d_old, precondition=True):
    """Do different variants of the conjugate gradient method.

    Reference: https://indrag49.github.io/Numerical-Optimization/conjugate-gradient-methods-1.html

    Args:
        scf: SCF object.
        ik (int): k-point index.
        cgform (int): Conjugate gradient form.
        g (ndarray): Current gradient.
        g_old (ndarray): Previous gradient.
        d_old (ndarray): Previous search direction.

    Keyword Args:
        precondition (bool): Weather to use a preconditioner.

    Returns:
        tuple[float, float]: Conjugate scalar and gradient norm.
    """
    atoms = scf.atoms

    if precondition:
        Kg, Kg_old = atoms.K(g, ik), atoms.K(g_old, ik)
    else:
        Kg, Kg_old = g, g_old
    norm_g = dotprod(g, Kg)

    if cgform == 1:    # Fletcher-Reeves
        return norm_g / dotprod(g_old, Kg_old), norm_g
    if cgform == 2:  # Polak-Ribiere
        return dotprod(g - g_old, Kg) / dotprod(g_old, Kg_old), norm_g
    if cgform == 3:  # Hestenes-Stiefel
        return dotprod(g - g_old, Kg) / dotprod(g - g_old, d_old), norm_g
    if cgform == 4:  # Dai-Yuan
        return norm_g / dotprod(g - g_old, d_old), norm_g
    ValueError(f'No cgform found for "{cgform}".')
    return None


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

    if precondition:
        method = 'pclm'
    else:
        method = 'lm'

    # Scalars that need to be saved for each spin
    linmin = np.empty((len(atoms.wk), atoms.occ.Nspin))
    beta = np.empty((len(atoms.wk), atoms.occ.Nspin, 1, 1))
    # Search direction that needs to be saved for each spin
    d = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]

    for _ in range(Nit):
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin each spin separately
                if scf.log.level <= logging.DEBUG and Nit > 0:
                    linmin[ik][spin] = linmin_test(g, d[ik][spin])
                if precondition:
                    d[ik][spin] = -atoms.K(g, ik)
                else:
                    d[ik][spin] = -g
                W_tmp = copy.deepcopy(scf.W)
                W_tmp[ik] = scf.W[ik] + betat * d[ik][spin]
                gt = grad(scf, ik, spin, W_tmp)
                beta[ik][spin] = betat * dotprod(g, d[ik][spin]) / dotprod(g - gt, d[ik][spin])

        for ik in range(len(atoms.wk)):
            scf.W[ik] = scf.W[ik] + beta[ik] * d[ik]
        c = cost(scf)
        costs.append(c)
        if condition(scf, method, costs, linmin):
            break
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

    if precondition:
        method = 'pccg'
    else:
        method = 'cg'

    # Scalars that need to be saved for each spin
    linmin = np.empty((len(atoms.wk), atoms.occ.Nspin))
    cg = np.empty((len(atoms.wk), atoms.occ.Nspin))
    beta = np.empty((len(atoms.wk), atoms.occ.Nspin, 1, 1))
    norm_g = np.empty((len(atoms.wk), atoms.occ.Nspin))
    # Gradients that need to be saved for each spin
    d = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    d_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    g_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]

    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(len(atoms.wk)):
        for spin in range(atoms.occ.Nspin):
            g = grad(scf, ik, spin, scf.W)
            if precondition:
                d[ik][spin] = -atoms.K(g, ik)
            else:
                d[ik][spin] = -g
            W_tmp = copy.deepcopy(scf.W)
            W_tmp[ik] = scf.W[ik] + betat * d[ik][spin]
            gt = grad(scf, ik, spin, W_tmp)
            beta[ik][spin] = betat * dotprod(g, d[ik][spin]) / dotprod(g - gt, d[ik][spin])
            g_old[ik][spin], d_old[ik][spin] = g, d[ik][spin]

    # Update wave functions after calculating the gradients for each spin
    for ik in range(len(atoms.wk)):
        scf.W[ik] = scf.W[ik] + beta[ik] * d[ik]
    c = cost(scf)
    costs.append(c)
    condition(scf, method, costs)

    for _ in range(1, Nit):
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf.log.level <= logging.DEBUG:
                    linmin[ik][spin] = linmin_test(g, d[ik][spin])
                    cg[ik][spin] = cg_test(atoms, g, g_old[ik][spin], precondition)
                beta[ik][spin], norm_g[ik][spin] = cg_method(scf, ik, cgform, g, g_old[ik][spin], d_old[ik][spin],
                                                    precondition)
                if precondition:
                    d[ik][spin] = -atoms.K(g, ik) + beta[ik][spin] * d_old[ik][spin]
                else:
                    d[ik][spin] = -g + beta[ik][spin] * d_old[ik][spin]
                W_tmp = copy.deepcopy(scf.W)
                W_tmp[ik] = scf.W[ik] + betat * d[ik][spin]
                gt = grad(scf, ik, spin, W_tmp)
                beta[ik][spin] = betat * dotprod(g, d[ik][spin]) / dotprod(g - gt, d[ik][spin])
                g_old[ik][spin], d_old[ik][spin] = g, d[ik][spin]

        for ik in range(len(atoms.wk)):
            scf.W[ik] = scf.W[ik] + beta[ik] * d[ik]
        c = cost(scf)
        costs.append(c)
        if condition(scf, method, costs, linmin, cg, norm_g):
            break
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

    # Scalars that need to be saved for each spin
    linmin = np.empty((len(atoms.wk), atoms.occ.Nspin))
    cg = np.empty((len(atoms.wk), atoms.occ.Nspin))
    beta = np.empty((len(atoms.wk), atoms.occ.Nspin, 1, 1))
    norm_g = np.empty((len(atoms.wk), atoms.occ.Nspin))
    # Gradients that need to be saved for each spin
    g = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    d = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    d_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]
    g_old = [np.empty_like(scf.W[ik], dtype=complex) for ik in range(len(atoms.wk))]

    # Do the first step without the linmin and cg tests, and without the cg_method
    for ik in range(len(atoms.wk)):
        for spin in range(atoms.occ.Nspin):
            g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
            d[ik][spin] = -atoms.K(g[ik][spin], ik)
            W_tmp = copy.deepcopy(scf.W)
            W_tmp[ik] = scf.W[ik] + betat * d[ik][spin]
            gt = grad(scf, ik, spin, W_tmp)
            beta[ik][spin] = betat * dotprod(g[ik][spin], d[ik][spin]) / dotprod(g[ik][spin] - gt, d[ik][spin])
            g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]

    # Update wave functions after calculating the gradients for each spin, save the wave function
    W_old = copy.deepcopy(scf.W)
    for ik in range(len(atoms.wk)):
        scf.W[ik] = scf.W[ik] + beta[ik] * d[ik]
    c = cost(scf)
    costs.append(c)
    if condition(scf, 'pccg', costs):
        return costs

    for _ in range(1, Nit):
        for ik in range(len(atoms.wk)):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf.log.level <= logging.DEBUG:
                    linmin[ik][spin] = linmin_test(g, d[ik][spin])
                    cg[ik][spin] = cg_test(atoms, g, g_old[ik][spin])
                beta[ik][spin], norm_g[ik][spin] = cg_method(scf, ik, cgform, g[ik][spin], g_old[ik][spin], d_old[ik][spin])
                d[ik][spin] = -atoms.K(g[ik][spin], ik) + beta[ik][spin] * d_old[ik][spin]
                W_tmp = copy.deepcopy(scf.W)
                W_tmp[ik] = scf.W[ik] + betat * d[ik][spin]
                gt = grad(scf, ik, spin, W_tmp)
                beta[ik][spin] = betat * dotprod(g[ik][spin], d[ik][spin]) / dotprod(g[ik][spin] - gt, d[ik][spin])
                g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]

        W_old = copy.deepcopy(scf.W)
        for ik in range(len(atoms.wk)):
            scf.W[ik] = scf.W[ik] + beta[ik] * d[ik]
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
