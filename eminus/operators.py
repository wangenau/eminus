#!/usr/bin/env python3
'''Basis set dependent operators for a plane-wave basis.'''
from os import environ

import numpy as np
from scipy.fft import fftn, ifftn

try:
    THREADS = int(environ['OMP_NUM_THREADS'])
except KeyError:
    THREADS = None


def O(atoms, W):
    '''Overlap operator.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    return atoms.Omega * W


def L(atoms, W):
    '''Laplacian operator.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    # G2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if len(W) == len(atoms.G2c):
        G2 = atoms.G2c.reshape(-1, 1)
    else:
        G2 = atoms.G2.reshape(-1, 1)

    return -atoms.Omega * G2 * W


def Linv(atoms, W):
    '''Inverse Laplacian operator.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    out = np.empty_like(W, dtype=complex)
    # Ignore the division by zero for the first elements
    # One could do some proper indexing with [1:], but this version is way faster
    with np.errstate(divide='ignore', invalid='ignore'):
        if W.ndim == 1:
            out = W / atoms.G2 / -atoms.Omega
            out[0] = 0
        else:
            G2 = atoms.G2.reshape(-1, 1)
            out = W / G2 / -atoms.Omega
            out[0, :] = 0
    return out


def K(atoms, W):
    '''Preconditioning operator. Applies 1/(1+G2) to the input.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    # G2 is a normal 1d row vector, reshape it so it can be applied to the column vector W
    if len(W) == len(atoms.G2c):
        G2 = atoms.G2c.reshape(-1, 1)
    else:
        G2 = atoms.G2.reshape(-1, 1)

    return W / (1 + G2)


def I(atoms, W):
    '''Backwards transformation from reciprocal space to real-space.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    W = W.T
    if W.ndim == 1:
        if len(W) == np.prod(atoms.s):
            tmp = np.reshape(W, atoms.s, order='F')
            Finv = ifftn(tmp, workers=THREADS).flatten(order='F')
        else:
            full = np.zeros(np.prod(atoms.s), dtype=complex)
            full[atoms.active] = W
            full = np.reshape(full, atoms.s, order='F')
            Finv = ifftn(full, workers=THREADS).flatten(order='F')
    elif W.shape[1] == np.prod(atoms.s):
        Finv = np.empty_like(W, dtype=complex)
        for i in range(len(W)):
            tmp = np.reshape(W[i], atoms.s, order='F')
            Finv[i] = ifftn(tmp, workers=THREADS).flatten(order='F')
    else:
        Finv = np.empty((len(W), np.prod(atoms.s)), dtype=complex)
        for i in range(len(W)):
            full = np.zeros(np.prod(atoms.s), dtype=complex)
            full[atoms.active] = W[i]
            full = np.reshape(full, atoms.s, order='F')
            Finv[i] = ifftn(full, workers=THREADS).flatten(order='F')
    return (Finv * np.prod(atoms.s)).T


def J(atoms, W, full=True):
    '''Forward transformation from real-space to reciprocal space.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Keyword Args:
        full (bool): Wether to transform in the full or in the active space.

    Returns:
        array: The operator applied on W.
    '''
    W = W.T
    if W.ndim == 1:
        tmp = np.reshape(W, atoms.s, order='F')
        F = fftn(tmp, workers=THREADS).flatten(order='F')
        if not full:
            F = F[atoms.active]
    else:
        if full:
            F = np.empty_like(W, dtype=complex)
            for i in range(len(W)):
                tmp = np.reshape(W[i], atoms.s, order='F')
                F[i] = fftn(tmp, workers=THREADS).flatten(order='F')
        else:
            F = np.empty((len(W), len(atoms.G2c)), dtype=complex)
            for i in range(len(W)):
                tmp = np.reshape(W[i], atoms.s, order='F')
                F[i] = fftn(tmp, workers=THREADS).flatten(order='F')[atoms.active]
    return (F / np.prod(atoms.s)).T


def Idag(atoms, W):
    '''Conjugated backwards transformation from reciprocal space to real-space.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    W = W.T
    if W.ndim == 1:
        tmp = np.reshape(W, atoms.s, order='F')
        full = fftn(tmp, workers=THREADS).flatten(order='F')
        F = full[atoms.active]
    else:
        F = np.empty((len(W), len(atoms.G2c)), dtype=complex)
        for i in range(len(W)):
            tmp = np.reshape(W[i], atoms.s, order='F')
            full = fftn(tmp, workers=THREADS).flatten(order='F')
            F[i] = full[atoms.active]
    return F.T


def Jdag(atoms, W):
    '''Conjugated forward transformation from real-space to reciprocal space.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    W = W.T
    if W.ndim == 1:
        tmp = np.reshape(W, atoms.s, order='F')
        Finv = ifftn(tmp, workers=THREADS).flatten(order='F')
    else:
        Finv = np.empty_like(W, dtype=complex)
        for i in range(len(W)):
            tmp = np.reshape(W[i], atoms.s, order='F')
            Finv[i] = ifftn(tmp, workers=THREADS).flatten(order='F')
    return Finv.T


def T(atoms, W, dr):
    '''Translation operator. Shifts input orbitals by the vector dr.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.
        dr (array): Real-space shift.

    Returns:
        array: The operator applied on W.
    '''
    out = np.empty_like(W, dtype=complex)
    factor = np.exp(-1j * atoms.G[atoms.active] @ dr)
    for i in range(atoms.Ns):
        out[:, i] = factor * W[:, i]
    return out
