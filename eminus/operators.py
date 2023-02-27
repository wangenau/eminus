#!/usr/bin/env python3
'''Basis set dependent operators for a plane wave basis.

These operators act on discretized wave functions, i.e., the arrays W.

These W are column vectors. This has been chosen to let theory and code coincide, e.g.,
W^dagger W becomes W.conj().T @ W.

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
'''
import os

import numpy as np
from scipy.fft import fftn, ifftn

from .utils import handle_spin_gracefully

try:
    THREADS = int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    THREADS = None


# Spin handling is trivial for this operator
def O(atoms, W):
    '''Overlap operator.

    This operator acts on the options 3, 4, 5, and 6.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    '''
    return atoms.Omega * W


@handle_spin_gracefully
def L(atoms, W):
    '''Laplacian operator.

    This operator acts on the options 3 and 5.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    '''
    # G2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if len(W) == len(atoms.G2c):
        G2 = atoms.G2c[:, None]
    else:
        G2 = atoms.G2[:, None]
    return -atoms.Omega * G2 * W


@handle_spin_gracefully
def Linv(atoms, W):
    '''Inverse Laplacian operator.

    This operator acts on the options 3 and 4.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    '''
    out = np.empty_like(W, dtype=complex)
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
def I(atoms, W):
    '''Backwards transformation from reciprocal space to real-space.

    This operator acts on the options 3, 4, 5, and 6.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    '''
    n = np.prod(atoms.s)

    # If W is in the full space do nothing with W
    if len(W) == len(atoms.G2):
        Wfft = np.copy(W)
    else:
        # Fill with zeros if W is in the active space
        if W.ndim == 1:
            Wfft = np.zeros(n, dtype=complex)
        else:
            Wfft = np.zeros((n, atoms.Nstate), dtype=complex)
        Wfft[atoms.active] = W

    # `workers` sets the number of threads the FFT operates on
    # `overwrite_x` allows writing in Wfft, but since we do not need Wfft later on, we can set this
    # for a little bit of extra performance
    # Normally, we would have to multiply by n in the end for the correct normalization, but we can
    # ignore this step when properly setting the `norm` option for a faster operation
    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        Finv = ifftn(Wfft, workers=THREADS, overwrite_x=True, norm='forward').ravel()
    else:
        # Here we reshape the input like in the 1d case but add an extra dimension in the end,
        # holding the number of states
        Wfft = Wfft.reshape(np.append(atoms.s, atoms.Nstate))
        # Tell the function that the FFT only has to act on the first 3 axes
        Finv = ifftn(Wfft, workers=THREADS, overwrite_x=True, norm='forward',
                     axes=(0, 1, 2)).reshape((n, atoms.Nstate))
    return Finv


@handle_spin_gracefully
def J(atoms, W, full=True):
    '''Forward transformation from real-space to reciprocal space.

    This operator acts on the options 1 and 2.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        full (bool): Wether to transform in the full or in the active space.

    Returns:
        ndarray: The operator applied on W.
    '''
    n = np.prod(atoms.s)

    # `workers` sets the number of threads the FFT operates on
    # `overwrite_x` allows writing in Wfft, but since we do not need Wfft later on, we can set this
    # for a little bit of extra performance
    # Normally, we would have to divide by n in the end for the correct normalization, but we can
    # ignore this step when properly setting the `norm` option for a faster operation
    if W.ndim == 1:
        Wfft = W.reshape(atoms.s)
        F = fftn(Wfft, workers=THREADS, overwrite_x=True, norm='forward').ravel()
    else:
        Wfft = W.reshape(np.append(atoms.s, atoms.Nstate))
        F = fftn(Wfft, workers=THREADS, overwrite_x=True, norm='forward',
                 axes=(0, 1, 2)).reshape((n, atoms.Nstate))

    # There is no way to know if J has to transform to the full or the active space
    # but normally it transforms to the full space
    if not full:
        return F[atoms.active]
    return F


@handle_spin_gracefully
def Idag(atoms, W, full=False):
    '''Conjugated backwards transformation from real-space to reciprocal space.

    This operator acts on the options 1 and 2.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        full (bool): Wether to transform in the full or in the active space.

    Returns:
        ndarray: The operator applied on W.
    '''
    n = np.prod(atoms.s)
    F = J(atoms, W, full)
    return F * n


@handle_spin_gracefully
def Jdag(atoms, W):
    '''Conjugated forward transformation from reciprocal space to real-space.

    This operator acts on the options 3, 4, 5, and 6.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    '''
    n = np.prod(atoms.s)
    Finv = I(atoms, W)
    return Finv / n


@handle_spin_gracefully
def K(atoms, W):
    '''Preconditioning operator.

    This operator acts on the options 3 and 5.
    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    '''
    # G2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if len(W) == len(atoms.G2c):
        G2 = atoms.G2c[:, None]
    else:
        G2 = atoms.G2[:, None]
    return W / (1 + G2)


@handle_spin_gracefully
def T(atoms, W, dr):
    '''Translation operator.

    This operator acts on the options 5 and 6.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.
        dr (ndarray): Real-space shift.

    Returns:
        ndarray: The operator applied on W.
    '''
    # Do the shift by multiplying a phase factor, given by the shift theorem
    factor = np.exp(-1j * atoms.G[atoms.active] @ dr)
    # factor is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if W.ndim == 2:
        factor = factor[:, None]
    return factor * W
