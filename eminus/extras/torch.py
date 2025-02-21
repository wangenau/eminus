# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Implementation of operators using Torch FFT functions.

For more details see :mod:`~eminus.operators`.

For installation instructions see https://pytorch.org/get-started/locally.

Alternatively, one can try the default installation with::

    pip install eminus[torch]

This implementation is focused on speed, rather than readability since these operators need the
most time in most calculations. Notable differences to the default operators are:

* Use Torch FFTs (we need to cast atoms.s to tuples for this)
* No handle_spin decorator to reduce function calls and casts
* Call properties directly using their private attribute
* Explicitly implement Idag and Jdag for one function call less and direct normalization
* Optional GPU calculations

In my tests, the overhead to move the arrays to the GPU and back is not worth it, so it is disabled
by default.

Reference: Adv. Neural. Inf. Process Syst. 32, 8024.
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
    import torch

    n = atoms._Ns
    s = tuple(atoms._s)

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

    Wfft = torch.from_numpy(Wfft)
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        Finv = torch.fft.ifftn(Wfft, s=s, norm="forward").view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view((*s, W.shape[-1]))
        Finv = torch.fft.ifftn(Wfft, s=s, norm="forward", dim=(0, 1, 2)).view(n, W.shape[-1])
    else:
        Wfft = Wfft.view((atoms.occ._Nspin, *s, W.shape[-1]))
        Finv = torch.fft.ifftn(Wfft, s=s, norm="forward", dim=(1, 2, 3)).view(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    return Finv.detach().cpu().numpy()


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
    import torch

    n = atoms._Ns
    s = tuple(atoms._s)

    Wfft = torch.from_numpy(W)
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        F = torch.fft.fftn(Wfft, s=s, norm="forward").view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view((*s, W.shape[-1]))
        F = torch.fft.fftn(Wfft, s=s, norm="forward", dim=(0, 1, 2)).view(n, W.shape[-1])
    else:
        Wfft = Wfft.view((atoms.occ._Nspin, *s, W.shape[-1]))
        F = torch.fft.fftn(Wfft, s=s, norm="forward", dim=(1, 2, 3)).view(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    F = F.detach().cpu().numpy()
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
    import torch

    n = atoms._Ns
    s = tuple(atoms._s)

    Wfft = torch.from_numpy(W)
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        F = torch.fft.fftn(Wfft, s=s, norm="backward").view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view((*s, W.shape[-1]))
        F = torch.fft.fftn(Wfft, s=s, norm="backward", dim=(0, 1, 2)).view(n, W.shape[-1])
    else:
        Wfft = Wfft.view((atoms.occ._Nspin, *s, W.shape[-1]))
        F = torch.fft.fftn(Wfft, s=s, norm="backward", dim=(1, 2, 3)).view(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    F = F.detach().cpu().numpy()
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
    import torch

    n = atoms._Ns
    s = tuple(atoms._s)

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

    Wfft = torch.from_numpy(Wfft)
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        Finv = torch.fft.ifftn(Wfft, s=s, norm="backward").view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view((*s, W.shape[-1]))
        Finv = torch.fft.ifftn(Wfft, s=s, norm="backward", dim=(0, 1, 2)).view(n, W.shape[-1])
    else:
        Wfft = Wfft.view((atoms.occ._Nspin, *s, W.shape[-1]))
        Finv = torch.fft.ifftn(Wfft, s=s, norm="backward", dim=(1, 2, 3)).view(
            atoms.occ._Nspin, n, W.shape[-1]
        )
    return Finv.detach().cpu().numpy()
