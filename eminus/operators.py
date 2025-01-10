# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Basis set dependent operators for a plane wave basis.

These operators act on discretized wave functions, i.e., the arrays W.

These W are column vectors. This has been chosen to let theory and code coincide, e.g.,
W^dagger W becomes :code:`W.conj().T @ W`.

The downside is that the i-th state will be accessed with W[:, i] instead of W[i].
Choosing the i-th state makes the array 1d.

These operators can act on six different options, namely

1. the real-space
2. the real-space (1d)
3. the full reciprocal space
4. the full reciprocal space (1d)
5. the active reciprocal space
6. the active reciprocal space (1d)

The active space is the truncated reciprocal space by restricting it with a sphere given by ecut.

Every spin dependence will be handled with handle_spin by calling the operators for each
spin individually. The same goes for the handling of k-points, while for k-points W is represented
as a list of arrays. This gives the final indexing for the k-point k, spin s, and state n of
W[ik][s, :, n].
"""

import copy

import numpy as np
from scipy.fft import fftn, ifftn

from . import config
from .utils import handle_backend, handle_k, handle_spin


# Spin handling is trivial for this operator
@handle_k
def O(atoms, W):
    """Overlap operator.

    This operator acts on the options 3, 4, 5, and 6.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        The operator applied on W.
    """
    return atoms.Omega * W


@handle_spin
def L(atoms, W, ik=-1):
    """Laplacian operator with k-point dependency.

    This operator acts on options 3 and 5.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.

    Returns:
        The operator applied on W.
    """
    # Gk2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if len(W) == len(atoms.Gk2c[ik]):
        Gk2 = atoms.Gk2c[ik][:, None]
    else:
        Gk2 = atoms.Gk2[ik][:, None]
    return -atoms.Omega * Gk2 * W


@handle_spin
def Linv(atoms, W):
    """Inverse Laplacian operator.

    This operator acts on options 3 and 4.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        The operator applied on W.
    """
    # Ignore the division by zero for the first elements
    with np.errstate(divide="ignore", invalid="ignore"):
        if W.ndim == 1:
            # One could do some proper indexing with [1:] but indexing is slow
            out = W / (atoms.G2 * -atoms.Omega)
            out[0] = 0
        else:
            # G2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
            G2 = atoms.G2[:, None]
            out = W / (G2 * -atoms.Omega)
            out[0, :] = 0
    return out


@handle_backend
@handle_k(mode="index")
@handle_spin
def I(atoms, W, ik=-1):
    """Backward transformation from reciprocal space to real-space.

    This operator acts on the options 3, 4, 5, and 6.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.

    Returns:
        The operator applied on W.
    """
    n = atoms.Ns

    # If W is in the full space do nothing with W
    if len(W) == len(atoms.Gk2[ik]):
        Wfft = np.copy(W)
    else:
        # Fill with zeros if W is in the active space
        if W.ndim == 1:
            Wfft = np.zeros(n, dtype=W.dtype)
        else:
            Wfft = np.zeros((n, W.shape[-1]), dtype=W.dtype)
        Wfft[atoms.active[ik]] = W

    # `workers` sets the number of threads the FFT operates on
    # `overwrite_x` allows writing in Wfft, but since we do not need Wfft later on, we can set this
    # for a little bit of extra performance
    # Normally, we would have to multiply by n in the end for the correct normalization, but we can
    # ignore this step when properly setting the `norm` option for a faster operation
    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        Finv = ifftn(Wfft, workers=config.threads, overwrite_x=True, norm="forward").ravel()
    else:
        # Here we reshape the input like in the 1d case but add an extra dimension in the end,
        # holding the number of states
        Wfft = Wfft.reshape(np.append(atoms.s, W.shape[-1]))
        # Tell the function that the FFT only has to act on the first 3 axes
        Finv = ifftn(
            Wfft, workers=config.threads, overwrite_x=True, norm="forward", axes=(0, 1, 2)
        ).reshape((n, W.shape[-1]))
    return Finv


@handle_backend
@handle_k(mode="index")
@handle_spin
def J(atoms, W, ik=-1, full=True):
    """Forward transformation from real-space to reciprocal space.

    This operator acts on options 1 and 2.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.
        full: Whether to transform in the full or in the active space.

    Returns:
        The operator applied on W.
    """
    n = atoms.Ns
    Wfft = np.copy(W)

    # `workers` sets the number of threads the FFT operates on
    # `overwrite_x` allows writing in Wfft, but since we do not need Wfft later on, we can set this
    # for a little bit of extra performance
    # Normally, we would have to divide by n in the end for the correct normalization, but we can
    # ignore this step when properly setting the `norm` option for a faster operation
    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        F = fftn(Wfft, workers=config.threads, overwrite_x=True, norm="forward").ravel()
    else:
        Wfft = Wfft.reshape(np.append(atoms.s, W.shape[-1]))
        F = fftn(
            Wfft, workers=config.threads, overwrite_x=True, norm="forward", axes=(0, 1, 2)
        ).reshape((n, W.shape[-1]))

    # There is no way to know if J has to transform to the full or the active space
    # but normally it transforms to the full space
    if not full:
        return F[atoms.active[ik]]
    return F


@handle_backend
@handle_k(mode="index")
@handle_spin
def Idag(atoms, W, ik=-1, full=False):
    """Conjugated backward transformation from real-space to reciprocal space.

    This operator acts on options 1 and 2.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.
        full: Whether to transform in the full or in the active space.

    Returns:
        The operator applied on W.
    """
    n = atoms.Ns
    F = J(atoms, W, ik, full)
    return F * n


@handle_backend
@handle_k(mode="index")
@handle_spin
def Jdag(atoms, W, ik=-1):
    """Conjugated forward transformation from reciprocal space to real-space.

    This operator acts on the options 3, 4, 5, and 6.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.

    Returns:
        The operator applied on W.
    """
    n = atoms.Ns
    Finv = I(atoms, W, ik)
    return Finv / n


@handle_spin
def K(atoms, W, ik):
    """Preconditioning operator with k-point dependency.

    This operator acts on options 3 and 5.

    Reference: Comput. Mater. Sci. 14, 4.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik: k-point index.

    Returns:
        The operator applied on W.
    """
    # Gk2c is a normal 1d row vector, reshape it so it can be applied to the column vector W
    return W / (1 + atoms.Gk2c[ik][:, None])


def T(atoms, W, dr):
    """Translation operator.

    This operator acts on options 5 and 6.

    Reference: https://ccrma.stanford.edu/~jos/st/Shift_Theorem.html

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik: k-point index.
        dr: Real-space shifting vector.

    Returns:
        The operator applied on W.
    """
    # We can not use a fancy decorator for this operator, so handle it here
    if isinstance(W, np.ndarray) and W.ndim == 3:
        return np.asarray([T(atoms, Wspin, dr) for Wspin in W])

    if isinstance(W, np.ndarray):
        atoms.kpts._assert_gamma_only()
        if len(W) == len(atoms.Gk2c[0]):
            G = atoms.G[atoms.active[0]]
        elif len(W) == len(atoms.Gk2c[-1]):
            G = atoms.G[atoms.active[-1]]
        else:
            G = atoms.G
        factor = np.exp(-1j * G @ dr)
        if W.ndim == 2:
            factor = factor[:, None]
        return factor * W

    # If W is a list we have to account for k-points
    Wshift = copy.deepcopy(W)
    for ik in range(atoms.kpts.Nk):
        # Do the shift by multiplying a phase factor, given by the shift theorem
        if W[ik].shape[1] == len(atoms.Gk2c[ik]):
            Gk = atoms.G[atoms.active[ik]] + atoms.kpts.k[ik]
        else:
            Gk = atoms.G + atoms.kpts.k[ik]
        Wshift[ik] = np.exp(-1j * Gk @ dr)[:, None] * W[ik]
    return Wshift
