#!/usr/bin/env python3
"""Implementation of operators using Torch FFT functions.

For more details see :mod:`~eminus.operators`.

For installation instructions see https://pytorch.org/get-started/locally.

This implementation is focused on speed, rather than readability since these operators need the
most time in most calculations. Notable differences to the default operators are:

* Use Torch FFTs (we need to cast atoms.s to tuples for this)
* No handle_spin_gracefully decorator to reduce function calls and casts
* Call properties directly using their private attribute
* Explicitly implement Idag and Jdag for one function call less and direct normalization
* Optional GPU calculations

In my tests, the overhead to move the arrays to the GPU and back is not worth it, so it is disabled
by default.

Reference: Adv. Neural. Inf. Process Syst. 32, 8024.
"""
import numpy as np

from .. import config


def I(atoms, W):
    """Backwards transformation from reciprocal space to real-space.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    """
    import torch
    n = np.prod(atoms._s)
    s = tuple(atoms._s)

    if W.ndim < 3:
        if len(W) == len(atoms._G2):
            Wfft = W
        else:
            if W.ndim == 1:
                Wfft = np.zeros(n, dtype=W.dtype)
            else:
                Wfft = np.zeros((n, atoms.occ._Nstate), dtype=W.dtype)
            Wfft[atoms._active] = W
    else:
        if W.shape[1] == len(atoms._G2):
            Wfft = W
        else:
            Wfft = np.zeros((atoms.occ._Nspin, n, atoms.occ._Nstate), dtype=W.dtype)
            Wfft[:, atoms._active[0]] = W

    Wfft = torch.from_numpy(Wfft)
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        Finv = torch.fft.ifftn(Wfft, s=s, norm='forward').view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view(s + (atoms.occ._Nstate,))
        Finv = torch.fft.ifftn(Wfft, s=s, norm='forward', dim=(0, 1, 2)).view(n, atoms.occ._Nstate)
    else:
        Wfft = Wfft.view((atoms.occ._Nspin,) + s + (atoms.occ._Nstate,))
        Finv = torch.fft.ifftn(Wfft, s=s, norm='forward', dim=(1, 2, 3)).view(atoms.occ._Nspin, n,
                                                                              atoms.occ._Nstate)
    return Finv.detach().cpu().numpy()


def J(atoms, W, full=True):
    """Forward transformation from real-space to reciprocal space.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        full (bool): Wether to transform in the full or in the active space.

    Returns:
        ndarray: The operator applied on W.
    """
    import torch
    n = np.prod(atoms._s)
    s = tuple(atoms._s)

    Wfft = torch.from_numpy(np.copy(W))
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        F = torch.fft.fftn(Wfft, s=s, norm='forward').view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view(s + (atoms.occ._Nstate,))
        F = torch.fft.fftn(Wfft, s=s, norm='forward', dim=(0, 1, 2)).view(n, atoms.occ._Nstate)
    else:
        Wfft = Wfft.view((atoms.occ._Nspin,) + s + (atoms.occ._Nstate,))
        F = torch.fft.fftn(Wfft, s=s, norm='forward', dim=(1, 2, 3)).view(atoms.occ._Nspin, n,
                                                                          atoms.occ._Nstate)
    F = F.detach().cpu().numpy()
    if not full:
        if F.ndim < 3:
            return F[atoms._active]
        return F[:, atoms._active[0]]
    return F


def Idag(atoms, W, full=False):
    """Conjugated backwards transformation from real-space to reciprocal space.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        full (bool): Wether to transform in the full or in the active space.

    Returns:
        ndarray: The operator applied on W.
    """
    import torch
    n = np.prod(atoms._s)
    s = tuple(atoms._s)

    Wfft = torch.from_numpy(np.copy(W))
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        F = torch.fft.fftn(Wfft, s=s, norm='forward').view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view(s + (atoms.occ._Nstate,))
        F = torch.fft.fftn(Wfft, s=s, norm='forward', dim=(0, 1, 2)).view(n, atoms.occ._Nstate)
    else:
        Wfft = Wfft.view((atoms.occ._Nspin,) + s + (atoms.occ._Nstate,))
        F = torch.fft.fftn(Wfft, s=s, norm='forward', dim=(1, 2, 3)).view(atoms.occ._Nspin, n,
                                                                          atoms.occ._Nstate)
    F = F.detach().cpu().numpy() * n
    if not full:
        if F.ndim < 3:
            return F[atoms._active]
        return F[:, atoms._active[0]]
    return F


def Jdag(atoms, W):
    """Conjugated forward transformation from reciprocal space to real-space.

    Args:
        atoms: Atoms object.
        W (ndarray): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        ndarray: The operator applied on W.
    """
    import torch
    n = np.prod(atoms._s)
    s = tuple(atoms._s)

    if W.ndim < 3:
        if len(W) == len(atoms._G2):
            Wfft = W
        else:
            if W.ndim == 1:
                Wfft = np.zeros(n, dtype=W.dtype)
            else:
                Wfft = np.zeros((n, atoms.occ._Nstate), dtype=W.dtype)
            Wfft[atoms._active] = W
    else:
        if W.shape[1] == len(atoms._G2):
            Wfft = W
        else:
            Wfft = np.zeros((atoms.occ._Nspin, n, atoms.occ._Nstate), dtype=W.dtype)
            Wfft[:, atoms._active[0]] = W

    Wfft = torch.from_numpy(Wfft)
    if config.use_gpu:
        Wfft = Wfft.cuda()

    if W.ndim == 1:
        Wfft = Wfft.view(s)
        Finv = torch.fft.ifftn(Wfft, s=s, norm='forward').view(n)
    elif W.ndim == 2:
        Wfft = Wfft.view(s + (atoms.occ._Nstate,))
        Finv = torch.fft.ifftn(Wfft, s=s, norm='forward', dim=(0, 1, 2)).view(n, atoms.occ._Nstate)
    else:
        Wfft = Wfft.view((atoms.occ._Nspin,) + s + (atoms.occ._Nstate,))
        Finv = torch.fft.ifftn(Wfft, s=s, norm='forward', dim=(1, 2, 3)).view(atoms.occ._Nspin, n,
                                                                              atoms.occ._Nstate)
    return Finv.detach().cpu().numpy() / n
