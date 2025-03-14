# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Implementation of operators using Jax FFT functions.

For more details see :mod:`~eminus.operators`.

For installation instructions see https://jax.readthedocs.io/en/latest/installation.html.

Alternatively, one can try the default installation with::

    pip install eminus[jax]

This implementation is focused on speed, rather than readability since these operators need the
most time in most calculations, see :mod:`~eminus.extras.torch`.

Note that in testings this backend was always slower than the torch or scipy backend.

Reference: https://github.com/google/jax.
"""

import numpy as np

from .. import config
from ..utils import handle_k


@handle_k(mode="index")
def I(atoms, W, ik=-1):
    """Backwards transformation from reciprocal space to real-space.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.

    Returns:
        The operator applied on W.
    """
    import jax
    import jax.numpy as jnp

    n = atoms._Ns

    if W.ndim < 3:
        if len(W) == len(atoms._Gk2[ik]):
            Wfft = W
        else:
            if W.ndim == 1:
                Wfft = np.zeros(n, dtype=W.dtype)
            else:
                Wfft = np.zeros((n, W.shape[-1]), dtype=W.dtype)
            Wfft[atoms._active[ik]] = W
    elif W.shape[1] == len(atoms._G2):
        Wfft = W
    else:
        Wfft = np.zeros((atoms.occ._Nspin, n, W.shape[-1]), dtype=W.dtype)
        Wfft[:, atoms._active[ik][0]] = W

    Wfft = jnp.asarray(Wfft)
    if config.use_gpu:
        gpus = jax.devices("gpu")
        Wfft = jax.device_put(Wfft, gpus[0])

    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        Finv = jnp.fft.ifftn(Wfft, norm="forward").ravel()
    elif W.ndim == 2:
        Wfft = Wfft.reshape((*atoms.s, W.shape[-1]))
        Finv = jnp.fft.ifftn(Wfft, norm="forward", axes=(0, 1, 2)).reshape(n, W.shape[-1])
    else:
        Wfft = Wfft.reshape((atoms.occ._Nspin, *atoms.s, W.shape[-1]))
        Finv = jnp.fft.ifftn(Wfft, norm="forward", axes=(1, 2, 3)).reshape(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    return np.asarray(Finv)


@handle_k(mode="index")
def J(atoms, W, ik=-1, full=True):
    """Forward transformation from real-space to reciprocal space.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.
        full: Whether to transform in the full or in the active space.

    Returns:
        The operator applied on W.
    """
    import jax
    import jax.numpy as jnp

    n = atoms._Ns

    Wfft = jnp.asarray(W)
    if config.use_gpu:
        gpus = jax.devices("gpu")
        Wfft = jax.device_put(Wfft, gpus[0])

    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        F = jnp.fft.fftn(Wfft, norm="forward").ravel()
    elif W.ndim == 2:
        Wfft = Wfft.reshape((*atoms.s, W.shape[-1]))
        F = jnp.fft.fftn(Wfft, norm="forward", axes=(0, 1, 2)).reshape(n, W.shape[-1])
    else:
        Wfft = Wfft.reshape((atoms.occ._Nspin, *atoms.s, W.shape[-1]))
        F = jnp.fft.fftn(Wfft, norm="forward", axes=(1, 2, 3)).reshape(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    F = np.asarray(F)
    if not full:
        if F.ndim < 3:
            return F[atoms._active[ik]]
        return F[:, atoms._active[ik][0]]
    return F


@handle_k(mode="index")
def Idag(atoms, W, ik=-1, full=False):
    """Conjugated backwards transformation from real-space to reciprocal space.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.
        full: Whether to transform in the full or in the active space.

    Returns:
        The operator applied on W.
    """
    import jax
    import jax.numpy as jnp

    n = atoms._Ns

    Wfft = jnp.asarray(W)
    if config.use_gpu:
        gpus = jax.devices("gpu")
        Wfft = jax.device_put(Wfft, gpus[0])

    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        F = jnp.fft.fftn(Wfft, norm="backward").ravel()
    elif W.ndim == 2:
        Wfft = Wfft.reshape((*atoms.s, W.shape[-1]))
        F = jnp.fft.fftn(Wfft, norm="backward", axes=(0, 1, 2)).reshape(n, W.shape[-1])
    else:
        Wfft = Wfft.reshape((atoms.occ._Nspin, *atoms.s, W.shape[-1]))
        F = jnp.fft.fftn(Wfft, norm="backward", axes=(1, 2, 3)).reshape(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    F = np.asarray(F)
    if not full:
        if F.ndim < 3:
            return F[atoms._active[ik]]
        return F[:, atoms._active[ik][0]]
    return F


@handle_k(mode="index")
def Jdag(atoms, W, ik=-1):
    """Conjugated forward transformation from reciprocal space to real-space.

    Args:
        atoms: Atoms object.
        W: Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        ik: k-point index.

    Returns:
        The operator applied on W.
    """
    import jax
    import jax.numpy as jnp

    n = atoms._Ns

    if W.ndim < 3:
        if len(W) == len(atoms._Gk2[ik]):
            Wfft = W
        else:
            if W.ndim == 1:
                Wfft = np.zeros(n, dtype=W.dtype)
            else:
                Wfft = np.zeros((n, W.shape[-1]), dtype=W.dtype)
            Wfft[atoms._active[ik]] = W
    elif W.shape[1] == len(atoms._G2):
        Wfft = W
    else:
        Wfft = np.zeros((atoms.occ._Nspin, n, W.shape[-1]), dtype=W.dtype)
        Wfft[:, atoms._active[ik][0]] = W

    Wfft = jnp.asarray(Wfft)
    if config.use_gpu:
        gpus = jax.devices("gpu")
        Wfft = jax.device_put(Wfft, gpus[0])

    if W.ndim == 1:
        Wfft = Wfft.reshape(atoms.s)
        Finv = jnp.fft.ifftn(Wfft, norm="backward").ravel()
    elif W.ndim == 2:
        Wfft = Wfft.reshape((*atoms.s, W.shape[-1]))
        Finv = jnp.fft.ifftn(Wfft, norm="backward", axes=(0, 1, 2)).reshape(n, W.shape[-1])
    else:
        Wfft = Wfft.reshape((atoms.occ._Nspin, *atoms.s, W.shape[-1]))
        Finv = jnp.fft.ifftn(Wfft, norm="backward", axes=(1, 2, 3)).reshape(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    return np.asarray(Finv)
