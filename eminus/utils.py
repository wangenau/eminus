#!/usr/bin/env python3
"""Linear algebra calculation utilities."""
import functools

import numpy as np
from scipy.linalg import norm

from .logger import log


def dotprod(a, b):
    """Efficiently calculate the expression a * b.

    Add an extra check to make sure the result is never zero since this function is used as a
    denominator in minimizers.

    Args:
        a (ndarray): Array of vectors.
        b (ndarray): Array of vectors.

    Returns:
        float: The expressions result
    """
    eps = 1e-15  # 2.22e-16 is the range of float64 machine precision
    # The dot product of complex vectors looks like the expression below, but this is slow
    # res = np.real(np.trace(a.conj().T @ b))
    # We can calculate the trace faster by taking the sum of the Hadamard product
    res = np.sum(a.conj() * b)
    if abs(res) < eps:
        return eps
    return np.real(res)


def Ylm_real(l, m, G):  # noqa: C901
    """Calculate real spherical harmonics from cartesian coordinates.

    Reference: https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/

    Args:
        l (int): Angular momentum number.
        m (int): Magnetic quantum number.
        G (ndarray): Recipocal lattice vector or array of lattice vectors.

    Returns:
        ndarray: Real spherical harmonics.
    """
    eps = 1e-9
    # Account for single vectors
    G = np.atleast_2d(G)

    # No need to calculate more for l=0
    if l == 0:
        return 0.5 * np.sqrt(1 / np.pi) * np.ones(len(G))

    # cos(theta)=Gz/|G|
    Gm = norm(G, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = G[:, 2] / Gm
    # Account for small magnitudes, if norm(G) < eps: cos_theta=0
    cos_theta[Gm < eps] = 0

    # Vectorized version of sin(theta)=sqrt(max(0, 1-cos_theta^2))
    sin_theta = np.sqrt(np.amax((np.zeros_like(cos_theta), 1 - cos_theta**2), axis=0))

    # phi=arctan(Gy/Gx)
    phi = np.arctan2(G[:, 1], G[:, 0])
    # If Gx=0: phi=pi/2*sign(Gy)
    phi_idx = np.abs(G[:, 0]) < eps
    phi[phi_idx] = np.pi / 2 * np.sign(G[phi_idx, 1])

    if l == 1:
        if m == -1:   # py
            return 0.5 * np.sqrt(3 / np.pi) * sin_theta * np.sin(phi)
        if m == 0:  # pz
            return 0.5 * np.sqrt(3 / np.pi) * cos_theta
        if m == 1:  # px
            return 0.5 * np.sqrt(3 / np.pi) * sin_theta * np.cos(phi)
    elif l == 2:
        if m == -2:    # dxy
            return np.sqrt(15 / 16 / np.pi) * sin_theta**2 * np.sin(2 * phi)
        if m == -1:  # dyz
            return np.sqrt(15 / 4 / np.pi) * cos_theta * sin_theta * np.sin(phi)
        if m == 0:   # dz2
            return 0.25 * np.sqrt(5 / np.pi) * (3 * cos_theta**2 - 1)
        if m == 1:   # dxz
            return np.sqrt(15 / 4 / np.pi) * cos_theta * sin_theta * np.cos(phi)
        if m == 2:   # dx2-y2
            return np.sqrt(15 / 16 / np.pi) * sin_theta**2 * np.cos(2 * phi)
    elif l == 3:
        if m == -3:
            return 0.25 * np.sqrt(35 / 2 / np.pi) * sin_theta**3 * np.sin(3 * phi)
        if m == -2:
            return 0.25 * np.sqrt(105 / np.pi) * sin_theta**2 * cos_theta * np.sin(2 * phi)
        if m == -1:
            return 0.25 * np.sqrt(21 / 2 / np.pi) * sin_theta * (5 * cos_theta**2 - 1) * np.sin(phi)
        if m == 0:
            return 0.25 * np.sqrt(7 / np.pi) * (5 * cos_theta**3 - 3 * cos_theta)
        if m == 1:
            return 0.25 * np.sqrt(21 / 2 / np.pi) * sin_theta * (5 * cos_theta**2 - 1) * np.cos(phi)
        if m == 2:
            return 0.25 * np.sqrt(105 / np.pi) * sin_theta**2 * cos_theta * np.cos(2 * phi)
        if m == 3:
            return 0.25 * np.sqrt(35 / 2 / np.pi) * sin_theta**3 * np.cos(3 * phi)

    log.error(f'No definition found for Ylm({l}, {m}).')
    return None


def handle_spin_gracefully(func):
    """Handle spin calculating the function for each channel seperately.

    This can only be applied if the only spin-dependent indexing is the wave function W.

    Implementing the explicit handling of spin adds an extra layer of complexity where one has to
    loop over the spin states in many places. We can hide this complexity using this decorator while
    still supporting many usecases, e.g., the operators previously act on arrays containing wave
    functions of all states and of one state only. This decorator maintains this functionality and
    adds the option to act on arrays containing wave functions of all spins and all states as well.

    Args:
        func (Callable): Function that acts on spin-states.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(object, W, *args, **kwargs):
        if W.ndim == 3:
            # If one is brave enough one could add multiprocessing over spin states right here
            return np.asarray([func(object, Wspin, *args, **kwargs) for Wspin in W])
        return func(object, W, *args, **kwargs)
    return decorator


def pseudo_uniform(size, seed=1234):
    """Lehmer random number generator, following MINSTD.

    Reference: Commun. ACM. 12, 85.

    Args:
        size (tuple): Dimension of the array to create.

    Keyword Args:
        seed (int): Seed to initialize the random number generator.

    Returns:
        ndarray: Array with (pseudo) random numbers.
    """
    W = np.zeros(size, dtype=complex)
    mult = 48271
    mod = (2**31) - 1
    x = (seed * mult + 1) % mod
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                x = (x * mult + 1) % mod
                W[i, j, k] = x / mod
    return W


def add_maybe_none(a, b):
    """Add a and b together, when one or both can potentially be None.

    Args:
        a (ndarray): Array.
        b (ndarray): Array.

    Returns:
        ndarray: Sum of a and b.
    """
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + b
