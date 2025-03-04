# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Minimization algorithms."""

import copy
import logging

import numpy as np

from .dft import get_epsilon, get_grad
from .energies import get_E, get_Eentropy
from .logger import name
from .utils import dotprod


def scf_step(scf, step):
    """Perform one SCF step for a DFT calculation.

    Calculating intermediate results speeds up the energy and gradient calculation.
    This function is similar to H_precompute but will set all variables and energies in the SCF
    class and returns the total energy.

    Args:
        scf: SCF object.
        step: Optimization step.

    Returns:
        Total energy.
    """
    scf.callback(scf, step)
    scf._precompute()
    # Update occupations every smear_update'th cycle
    if scf.atoms.occ.smearing > 0 and step % scf.smear_update == 0:
        epsilon = get_epsilon(scf, scf.W, **scf._precomputed)
        Efermi = scf.atoms.occ.smear(epsilon)
        get_Eentropy(scf, epsilon, Efermi)
    return get_E(scf)


def check_convergence(scf, method, Elist, linmin=None, cg=None, norm_g=None):
    """Check the energies for every SCF cycle and handle the output.

    Args:
        scf: SCF object.
        method: Minimization method.
        Elist: Total energies per SCF step.

    Keyword Args:
        linmin: Cosine between previous search direction and current gradient.
        cg: Conjugate-gradient orthogonality.
        norm_g: Gradient norm.

    Returns:
        Convergence condition.
    """
    iteration = len(Elist)

    # Print all the data
    print_scf_step(scf, method, Elist, linmin, cg, norm_g)

    if iteration > 1:
        # Check for convergence
        if scf.gradtol is None or norm_g is None:
            if abs(Elist[-1] - Elist[-2]) < scf.etol:
                scf.is_converged = True
                return True
        # If a gradient tolerance has been set we also check norm_g for convergence
        elif abs(Elist[-1] - Elist[-2]) < scf.etol and (np.sum(norm_g, axis=0) < scf.gradtol).all():
            scf.is_converged = True
            return True
        # Check if the current energy is higher than the last two values
        if (np.asarray(Elist[-3:-1]) < Elist[-1]).all():
            scf._log.warning("Total energy is not decreasing.")
    return False


def print_scf_step(scf, method, Elist, linmin, cg, norm_g):
    """Print the data of one SCF step and the header at the beginning.

    Args:
        scf: SCF object.
        method: Minimization method.
        Elist: Total energies per SCF step.
        linmin: Cosine between previous search direction and current gradient.
        cg: Conjugate-gradient orthogonality.
        norm_g: Gradient norm.
    """
    iteration = len(Elist)

    # Print a column header at the beginning
    # The ljust values just have been chosen such that the output looks decent
    if iteration == 1:
        header = "Method".ljust(8)
        header += "Iteration".ljust(11)
        header += "Etot [Eh]".ljust(13)
        header += "dEtot [Eh]".ljust(13)
        # Print the gradient norm for cg methods
        if method not in {"sd", "lm", "pclm"}:
            header += "|Gradient|".ljust(10 * scf.atoms.occ.Nspin + 3)
        # Print extra debugging information if available
        if scf._log.level <= logging.DEBUG:
            if method != "sd":
                header += "linmin-test".ljust(10 * scf.atoms.occ.Nspin + 3)
            if method not in {"sd", "lm", "pclm"}:
                header += "cg-test".ljust(10 * scf.atoms.occ.Nspin + 3)
            scf._log.debug(header)
        else:
            scf._log.info(header)

    # Print the information for every cycle
    # Context manager for printing norm_g, linmin, and cg
    with np.printoptions(formatter={"float": "{:+0.2e}".format}):
        info = f"{method:<8}{iteration:>8}   {Elist[-1]:<+13,.6f}"
        # In the first step we do not have all information yet
        if iteration > 1:
            info += f"{Elist[-1] - Elist[-2]:<+13,.4e}"
            if norm_g is not None:
                info += str(np.sum(norm_g, axis=0)).ljust(10 * scf.atoms.occ.Nspin + 3)
            if scf._log.level <= logging.DEBUG:
                if method != "sd" and linmin is not None:
                    info += str(np.sum(linmin, axis=0)).ljust(10 * scf.atoms.occ.Nspin + 3)
                if method not in {"sd", "lm", "pclm"} and cg is not None:
                    info += str(np.sum(cg, axis=0)).ljust(10 * scf.atoms.occ.Nspin + 3)
    if scf._log.level <= logging.DEBUG:
        scf._log.debug(info)
    else:
        scf._log.info(info)


def linmin_test(g, d):
    """Do the line minimization test.

    Calculate the cosine of the angle between g and d.

    Reference: https://trond.hjorteland.com/thesis/node26.html

    Args:
        g: Current gradient.
        d: Previous search direction.

    Returns:
        Linmin angle.
    """
    # cos = A B / |A| |B|
    return dotprod(g, d) / np.sqrt(dotprod(g, g) * dotprod(d, d))


def cg_test(atoms, ik, g, g_old, precondition=True):
    """Test the gradient-orthogonality theorem, i.e., g and g_old should be orthogonal.

    Calculate the cosine of the angle between g and g_old. For an angle of 90 degree the cosine goes
    to zero.

    Reference: https://math.uci.edu/~chenlong/CAMtips/CG.html

    Args:
        atoms: Atoms object.
        ik: k-point index.
        g: Current gradient.
        g_old: Previous gradient.

    Keyword Args:
        precondition: Whether to use a preconditioner.

    Returns:
        CG angle.
    """
    if precondition:
        Kg, Kg_old = atoms.K(g, ik), atoms.K(g_old, ik)
    else:
        Kg, Kg_old = g, g_old
    # cos = A B / |A| |B|
    return dotprod(g, Kg_old) / np.sqrt(dotprod(g, Kg) * dotprod(g_old, Kg_old))


def cg_method(scf, ik, cgform, g, g_old, d_old, precondition=True):
    """Do different variants of the conjugate gradient method.

    Reference: https://indrag49.github.io/Numerical-Optimization/conjugate-gradient-methods-1.html

    Args:
        scf: SCF object.
        ik: k-point index.
        cgform: Conjugate gradient form.
        g: Current gradient.
        g_old: Previous gradient.
        d_old: Previous search direction.

    Keyword Args:
        precondition: Whether to use a preconditioner.

    Returns:
        Conjugate scalar and gradient norm.
    """
    atoms = scf.atoms

    if precondition:
        Kg, Kg_old = atoms.K(g, ik), atoms.K(g_old, ik)
    else:
        Kg, Kg_old = g, g_old
    norm_g = dotprod(g, Kg)

    if cgform == 1:  # Fletcher-Reeves
        return norm_g / dotprod(g_old, Kg_old), norm_g
    if cgform == 2:  # Polak-Ribiere
        return dotprod(g - g_old, Kg) / dotprod(g_old, Kg_old), norm_g
    if cgform == 3:  # Hestenes-Stiefel
        return dotprod(g - g_old, Kg) / dotprod(g - g_old, d_old), norm_g
    if cgform == 4:  # Dai-Yuan
        return norm_g / dotprod(g - g_old, d_old), norm_g
    msg = f'No cgform found for "{cgform}".'
    raise ValueError(msg)


@name("steepest descent minimization")
def sd(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, **kwargs):
    """Steepest descent minimization algorithm.

    Args:
        scf: SCF object.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        **kwargs: Throwaway arguments.

    Returns:
        Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    for i in range(Nit):
        c = cost(scf, i)
        costs.append(c)
        if condition(scf, "sd", costs):
            break
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g = grad(scf, ik, spin, scf.W, **scf._precomputed)
                scf.W[ik][spin] = scf.W[ik][spin] - betat * g
    return costs


@name("preconditioned line minimization")
def pclm(
    scf,
    Nit,
    cost=scf_step,
    grad=get_grad,
    condition=check_convergence,
    betat=3e-5,
    precondition=True,
    **kwargs,
):
    """Preconditioned line minimization algorithm.

    Args:
        scf: SCF object.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        precondition: Whether to use a preconditioner.
        **kwargs: Throwaway arguments.

    Returns:
        Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    if precondition:
        method = "pclm"
    else:
        method = "lm"

    # Scalars that need to be saved for each spin
    linmin = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    # Search direction that needs to be saved for each spin
    d = [np.empty_like(Wk) for Wk in scf.W]
    g = [np.empty_like(Wk) for Wk in scf.W]

    for i in range(Nit):
        W_tmp = copy.deepcopy(scf.W)
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin each spin separately
                if scf._log.level <= logging.DEBUG and i > 0:
                    linmin[ik][spin] = linmin_test(g[ik][spin], d[ik][spin])
                if precondition:
                    d[ik][spin] = -atoms.K(g[ik][spin], ik)
                else:
                    d[ik][spin] = -g[ik][spin]
                scf.W[ik][spin] = scf.W[ik][spin] + betat * d[ik][spin]

        scf._precompute()
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
                beta = abs(
                    betat
                    * dotprod(g[ik][spin], d[ik][spin])
                    / dotprod(g[ik][spin] - gt, d[ik][spin])
                )
                scf.W[ik][spin] = W_tmp[ik][spin] + beta * d[ik][spin]
        c = cost(scf, i)
        costs.append(c)
        if condition(scf, method, costs, linmin):
            break
    return costs


@name("line minimization")
def lm(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, **kwargs):
    """Line minimization algorithm.

    Args:
        scf: SCF object.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        **kwargs: Throwaway arguments.

    Returns:
        Total energies per SCF cycle.
    """
    return pclm(scf, Nit, cost, grad, condition, betat, precondition=False)


@name("preconditioned conjugate-gradient minimization")
def pccg(
    scf,
    Nit,
    cost=scf_step,
    grad=get_grad,
    condition=check_convergence,
    betat=3e-5,
    cgform=1,
    precondition=True,
):
    """Preconditioned conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        cgform: Conjugate gradient form.
        precondition: Whether to use a preconditioner.

    Returns:
        Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    if precondition:
        method = "pccg"
    else:
        method = "cg"

    # Scalars that need to be saved for each spin and k-point
    linmin = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    cg = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    norm_g = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    # Gradients that need to be saved for each spin and k-point
    d = [np.empty_like(Wk) for Wk in scf.W]
    g = [np.empty_like(Wk) for Wk in scf.W]
    d_old = [np.empty_like(Wk) for Wk in scf.W]
    g_old = [np.empty_like(Wk) for Wk in scf.W]

    # Do the first step without the linmin and cg tests, and without the cg_method
    W_tmp = copy.deepcopy(scf.W)
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
            if precondition:
                d[ik][spin] = -atoms.K(g[ik][spin], ik)
            else:
                d[ik][spin] = -g[ik][spin]
            scf.W[ik][spin] = scf.W[ik][spin] + betat * d[ik][spin]

    # Calculate the optimal step width
    scf._precompute()
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
            beta = abs(
                betat * dotprod(g[ik][spin], d[ik][spin]) / dotprod(g[ik][spin] - gt, d[ik][spin])
            )
            scf.W[ik][spin] = W_tmp[ik][spin] + beta * d[ik][spin]
            g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]

    # Evaluate the cost function
    c = cost(scf, -1)
    costs.append(c)
    condition(scf, method, costs)

    # Start the iteration
    for i in range(1, Nit):
        W_tmp = copy.deepcopy(scf.W)
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin and cg for each spin and k-point separately if needed
                if scf._log.level <= logging.DEBUG:
                    linmin[ik][spin] = linmin_test(g[ik][spin], d[ik][spin])
                    cg[ik][spin] = cg_test(atoms, ik, g[ik][spin], g_old[ik][spin], precondition)
                beta, norm_g[ik][spin] = cg_method(
                    scf, ik, cgform, g[ik][spin], g_old[ik][spin], d_old[ik][spin], precondition
                )
                if precondition:
                    d[ik][spin] = -atoms.K(g[ik][spin], ik) + beta * d_old[ik][spin]
                else:
                    d[ik][spin] = -g[ik][spin] + beta * d_old[ik][spin]
                scf.W[ik][spin] = scf.W[ik][spin] + betat * d[ik][spin]

        scf._precompute()
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
                beta = abs(
                    betat
                    * dotprod(g[ik][spin], d[ik][spin])
                    / dotprod(g[ik][spin] - gt, d[ik][spin])
                )
                scf.W[ik][spin] = W_tmp[ik][spin] + beta * d[ik][spin]
                g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]

        c = cost(scf, i)
        costs.append(c)
        if condition(scf, method, costs, linmin, cg, norm_g):
            break
    return costs


@name("conjugate-gradient minimization")
def cg(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, cgform=1):
    """Conjugate-gradient minimization algorithm.

    Args:
        scf: SCF object.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        cgform: Conjugate gradient form.

    Returns:
        Total energies per SCF cycle.
    """
    return pccg(scf, Nit, cost, grad, condition, betat, cgform, precondition=False)


@name("auto minimization")
def auto(scf, Nit, cost=scf_step, grad=get_grad, condition=check_convergence, betat=3e-5, cgform=1):  # noqa: C901
    """Automatic preconditioned conjugate-gradient minimization algorithm.

    This function chooses an sd step over the pccg step if the energy goes up.

    Args:
        scf: SCF object.
        Nit: Maximum number of SCF steps.

    Keyword Args:
        cost: Function that will run every SCF step.
        grad: Function that calculates the respective gradient.
        condition: Function to check and log the convergence condition.
        betat: Step size.
        cgform: Conjugate gradient form.

    Returns:
        Total energies per SCF cycle.
    """
    atoms = scf.atoms
    costs = []

    # Scalars that need to be saved for each spin
    linmin = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    cg = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    norm_g = np.empty((atoms.kpts.Nk, atoms.occ.Nspin))
    # Gradients that need to be saved for each spin
    d = [np.empty_like(Wk) for Wk in scf.W]
    g = [np.empty_like(Wk) for Wk in scf.W]
    d_old = [np.empty_like(Wk) for Wk in scf.W]
    g_old = [np.empty_like(Wk) for Wk in scf.W]

    # Do the first step without the linmin and cg tests, and without the cg_method
    W_tmp = copy.deepcopy(scf.W)
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
            d[ik][spin] = -atoms.K(g[ik][spin], ik)
            scf.W[ik][spin] = scf.W[ik][spin] + betat * d[ik][spin]

    # Calculate the optimal step width
    scf._precompute()
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
            beta = abs(
                betat * dotprod(g[ik][spin], d[ik][spin]) / dotprod(g[ik][spin] - gt, d[ik][spin])
            )
            scf.W[ik][spin] = W_tmp[ik][spin] + beta * d[ik][spin]
            g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]

    # Evaluate the cost function
    c = cost(scf, -1)
    costs.append(c)
    if condition(scf, "pccg", costs):
        return costs

    # Start the iteration
    for i in range(1, Nit):
        W_tmp = copy.deepcopy(scf.W)
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                g[ik][spin] = grad(scf, ik, spin, scf.W, **scf._precomputed)
                # Calculate linmin and cg for each spin separately
                if scf._log.level <= logging.DEBUG:
                    linmin[ik][spin] = linmin_test(g[ik][spin], d[ik][spin])
                    cg[ik][spin] = cg_test(atoms, ik, g[ik][spin], g_old[ik][spin])
                beta, norm_g[ik][spin] = cg_method(
                    scf, ik, cgform, g[ik][spin], g_old[ik][spin], d_old[ik][spin]
                )
                d[ik][spin] = -atoms.K(g[ik][spin], ik) + beta * d_old[ik][spin]
                scf.W[ik][spin] = scf.W[ik][spin] + betat * d[ik][spin]

        scf._precompute()
        for ik in range(atoms.kpts.Nk):
            for spin in range(atoms.occ.Nspin):
                gt = grad(scf, ik, spin, scf.W, **scf._precomputed)
                beta = abs(
                    betat
                    * dotprod(g[ik][spin], d[ik][spin])
                    / dotprod(g[ik][spin] - gt, d[ik][spin])
                )
                scf.W[ik][spin] = W_tmp[ik][spin] + beta * d[ik][spin]
                g_old[ik][spin], d_old[ik][spin] = g[ik][spin], d[ik][spin]

        c = cost(scf, i)
        # If the energy does not go down use the steepest descent step and recalculate the energy
        if c > costs[-1]:
            for ik in range(atoms.kpts.Nk):
                scf.W[ik] = W_tmp[ik] - betat * g[ik]
            c = cost(scf, -1)
            costs.append(c)
            # Do not print cg and linmin if we do the sd step
            if condition(scf, "sd", costs, norm_g=norm_g):
                break
        else:
            costs.append(c)
            if condition(scf, "pccg", costs, linmin, cg, norm_g):
                break
    return costs


#: Map minimizer names with their respective implementation.
IMPLEMENTED = {
    "sd": sd,
    "lm": lm,
    "pclm": pclm,
    "cg": cg,
    "pccg": pccg,
    "auto": auto,
}
