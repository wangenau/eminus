#!/usr/bin/env python3
'''Basis set dependent operators for a plane-wave basis.'''
from os import environ

import numpy as np
from scipy.fft import ifftn, fftn

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
    W = W.T
    if W.shape[1] == len(atoms.G2c):
        return (-atoms.Omega * atoms.G2c * W).T
    else:
        return (-atoms.Omega * atoms.G2 * W).T


def Linv(atoms, W):
    '''Inverse Laplacian operator.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    W = W.T
    out = np.zeros_like(W, dtype=complex)
    out[0] = 0
    if W.ndim == 1:
        out[1:] = W[1:] / atoms.G2[1:] / -atoms.Omega
    else:
        for i in range(len(W)):
            out[i][1:] = W[i][1:] / atoms.G2[1:] / -atoms.Omega
    return out.T


def K(atoms, W):
    '''Preconditioning operator. Applies 1/(1+G2) to the input.

    Args:
        atoms: Atoms object.
        W (array): Expansion coefficients of unconstrained wave functions in reciprocal space.

    Returns:
        array: The operator applied on W.
    '''
    W = W.T
    out = np.empty_like(W, dtype=complex)
    if W.shape[1] == len(atoms.G2c):
        if W.ndim == 1:
            out = W / (1 + atoms.G2c)
        else:
            for i in range(len(W)):
                out[i] = W[i] / (1 + atoms.G2c)
    else:
        if W.ndim == 1:
            out = W / (1 + atoms.G2)
        else:
            for i in range(len(W)):
                out[i] = W[i] / (1 + atoms.G2)
    return out.T


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
        W = np.array([W])
    if np.size(W, 1) == np.prod(atoms.s):
        Finv = np.empty_like(W, dtype=complex)
        for i in range(W.shape[0]):
            tmp = np.reshape(W[i], atoms.s, order='F')
            Finv[i] = ifftn(tmp, workers=THREADS).flatten(order='F')
    else:
        Finv = np.empty((W.shape[0], np.prod(atoms.s)), dtype=complex)
        for i in range(W.shape[0]):
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
            for i in range(W.shape[0]):
                tmp = np.reshape(W[i], atoms.s, order='F')
                F[i] = fftn(tmp, workers=THREADS).flatten(order='F')
        else:
            F = np.empty((W.shape[0], len(atoms.active[0])), dtype=complex)
            for i in range(W.shape[0]):
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
        F = np.empty((np.size(W, 0), len(atoms.active[0])), dtype=complex)
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
    factor = np.exp(-1j * np.dot(atoms.Gc, dr))
    for i in range(atoms.Ns):
        out[:, i] = factor * W[:, i]
    return out
