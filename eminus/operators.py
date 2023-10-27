#!/usr/bin/env python3
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

Every spin dependence will be handled with handle_spin_gracefully by calling the operators for each
spin individually.
"""
import numpy as np
from scipy.fft import fftn, ifftn

from . import config
from .utils import handle_k_gracefully, handle_k_indexable, handle_spin_gracefully


# Spin handling is trivial for this operator
@handle_k_gracefully
def O(atoms, W):
    """Overlap operator.

    This operator acts on the options 3, 4, 5, and 6.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    """
    return atoms.Omega * W


@handle_spin_gracefully
def L(atoms, W, ik=0):
    """Laplacian operator with k-point dependency.

    This operator acts on options 3 and 5.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik (int): k-point index.

    Returns:
        ndarray: The operator applied on W.
    """
    # Gk2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if len(W) == len(atoms.Gk2c[ik]):
        Gk2 = atoms.Gk2c[ik][:, None]
    else:
        Gk2 = atoms.Gk2[ik][:, None]
    return -atoms.Omega * Gk2 * W


@handle_spin_gracefully
def Linv(atoms, W):
    """Inverse Laplacian operator.

    This operator acts on options 3 and 4.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    """
    # Ignore the division by zero for the first elements
    with np.errstate(divide='ignore', invalid='ignore'):
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


@handle_spin_gracefully
def I(atoms, W, ik=0):
    """Backward transformation from reciprocal space to real-space.

    This operator acts on the options 3, 4, 5, and 6.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik (int): k-point index.

    Returns:
        ndarray: The operator applied on W.
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
            Wfft = np.zeros((n, atoms.occ.Nstate), dtype=W.dtype)
        Wfft[atoms.active[ik]] = W

    # `workers` sets the number of threads the FFT operates on
    # `overwrite_x` allows writing in Wfft, but since we do not need Wfft later on, we can set this
    # for a little bit of extra performance
    # Normally, we would have to multiply by n in the end for the correct normalization, but we can
    # ignore this step when properly setting the `norm` option for a faster operation
    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        Finv = ifftn(Wfft, workers=config.threads, overwrite_x=True, norm='forward').ravel()
    else:
        # Here we reshape the input like in the 1d case but add an extra dimension in the end,
        # holding the number of states
        Wfft = Wfft.reshape(np.append(atoms.s, atoms.occ.Nstate))
        # Tell the function that the FFT only has to act on the first 3 axes
        Finv = ifftn(Wfft, workers=config.threads, overwrite_x=True, norm='forward',
                     axes=(0, 1, 2)).reshape((n, atoms.occ.Nstate))
    return Finv


@handle_spin_gracefully
def J(atoms, W, ik=0, full=True):
    """Forward transformation from real-space to reciprocal space.

    This operator acts on options 1 and 2.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik (int): k-point index.

    Keyword Args:
        full (bool): Whether to transform in the full or in the active space.

    Returns:
        ndarray: The operator applied on W.
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
        F = fftn(Wfft, workers=config.threads, overwrite_x=True, norm='forward').ravel()
    else:
        Wfft = Wfft.reshape(np.append(atoms.s, atoms.occ.Nstate))
        F = fftn(Wfft, workers=config.threads, overwrite_x=True, norm='forward',
                 axes=(0, 1, 2)).reshape((n, atoms.occ.Nstate))

    # There is no way to know if J has to transform to the full or the active space
    # but normally it transforms to the full space
    if not full:
        return F[atoms.active[ik]]
    return F


# Spin handling will be handled by the J operator
def Idag(atoms, W, ik=0, full=False):
    """Conjugated backward transformation from real-space to reciprocal space.

    This operator acts on options 1 and 2.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik (int): k-point index.

    Keyword Args:
        full (bool): Whether to transform in the full or in the active space.

    Returns:
        ndarray: The operator applied on W.
    """
    n = atoms.Ns
    F = J(atoms, W, ik, full)
    return F * n


# Spin handling will be handled by the I operator
def Jdag(atoms, W, ik=0):
    """Conjugated forward transformation from reciprocal space to real-space.

    This operator acts on the options 3, 4, 5, and 6.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik (int): k-point index.

    Returns:
        ndarray: The operator applied on W.
    """
    n = atoms.Ns
    Finv = I(atoms, W, ik)
    return Finv / n


@handle_k_indexable
@handle_spin_gracefully
def K(atoms, W, ik=0):
    """Preconditioning operator with k-point dependency.

    This operator acts on options 3 and 5.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        ik (int): k-point index.

    Returns:
        ndarray: The operator applied on W.
    """
    # Gk2c is a normal 1d row vector, reshape it so it can be applied to the column vector W
    return W / (1 + atoms.Gk2c[ik][:, None])


@handle_spin_gracefully
def T(atoms, W, dr):
    """Translation operator.

    This operator acts on options 5 and 6.

    Reference: https://ccrma.stanford.edu/~jos/st/Shift_Theorem.html

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        dr (ndarray): Real-space shifting vector.

    Returns:
        ndarray: The operator applied on W.
    """
    # Do the shift by multiplying a phase factor, given by the shift theorem
    if len(W) == len(atoms.G2c):
        G = atoms.G[atoms.active]
    else:
        G = atoms.G
    factor = np.exp(-1j * G @ dr)
    # factor is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if W.ndim == 2:
        factor = factor[:, None]
    return factor * W
