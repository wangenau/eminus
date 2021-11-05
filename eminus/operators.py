#!/usr/bin/env python3
'''
Basis set depedent operators for a plane-wave basis.
'''
from os import environ

import numpy as np
from scipy.fft import ifftn, fftn

try:
    THREADS = int(environ['OMP_NUM_THREADS'])
except KeyError:
    THREADS = None


def O(atoms, inp):
    '''Overlap operator.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    return atoms.CellVol * inp


def L(atoms, inp):
    '''Laplacian operator.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    if inp.shape[1] == len(atoms.G2c):
        return (-atoms.CellVol * atoms.G2c * inp).T
    else:
        return (-atoms.CellVol * atoms.G2 * inp).T


def Linv(atoms, inp):
    '''Inverse Laplacian operator.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    out = np.zeros_like(inp, dtype=complex)
    out[0] = 0
    if inp.ndim == 1:
        out[1:] = inp[1:] / atoms.G2[1:] / -atoms.CellVol
    else:
        for i in range(len(inp)):
            out[i][1:] = inp[i][1:] / atoms.G2[1:] / -atoms.CellVol
    return out.T


def K(atoms, inp):
    '''Preconditioning operator. Applies 1/(1+G2) to the input.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    out = np.empty_like(inp, dtype=complex)
    if inp.shape[1] == len(atoms.G2c):
        if inp.ndim == 1:
            out = inp / (1 + atoms.G2c)
        else:
            for i in range(len(inp)):
                out[i] = inp[i] / (1 + atoms.G2c)
    else:
        if inp.ndim == 1:
            out = inp / (1 + atoms.G2)
        else:
            for i in range(len(inp)):
                out[i] = inp[i] / (1 + atoms.G2)
    return out.T


def I(atoms, inp):
    '''Backwards transformation from reciprocal space to real-space.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    if inp.ndim == 1:
        inp = np.array([inp])
    if np.size(inp, 1) == np.prod(atoms.S):
        out = np.empty_like(inp, dtype=complex)
        for i in range(inp.shape[0]):
            tmp = np.reshape(inp[i], atoms.S, order='F')
            out[i] = ifftn(tmp, workers=THREADS).flatten(order='F')
    else:
        out = np.empty((inp.shape[0], np.prod(atoms.S)), dtype=complex)
        for i in range(inp.shape[0]):
            full = np.zeros(np.prod(atoms.S), dtype=complex)
            full[atoms.active] = inp[i]
            full = np.reshape(full, atoms.S, order='F')
            out[i] = ifftn(full, workers=THREADS).flatten(order='F')
    return (out * np.prod(atoms.S)).T


def J(atoms, inp, full=True):
    '''Forward transformation from real-space to reciprocal space.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Kwargs:
        full : bool
            Transform in the full or in the active/truncated space.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, atoms.S, order='F')
        out = fftn(tmp, workers=THREADS).flatten(order='F')
        if not full:
            out = out[atoms.active]
    else:
        if full:
            out = np.empty_like(inp, dtype=complex)
            for i in range(inp.shape[0]):
                tmp = np.reshape(inp[i], atoms.S, order='F')
                out[i] = fftn(tmp, workers=THREADS).flatten(order='F')
        else:
            out = np.empty((inp.shape[0], len(atoms.active[0])), dtype=complex)
            for i in range(inp.shape[0]):
                tmp = np.reshape(inp[i], atoms.S, order='F')
                out[i] = fftn(tmp, workers=THREADS).flatten(order='F')[atoms.active]
    return (out / np.prod(atoms.S)).T


def Idag(atoms, inp):
    '''Conjugated backwards transformation from reciprocal space to real-space.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, atoms.S, order='F')
        full = fftn(tmp, workers=THREADS).flatten(order='F')
        out = full[atoms.active]
    else:
        out = np.empty((np.size(inp, 0), len(atoms.active[0])), dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], atoms.S, order='F')
            full = fftn(tmp, workers=THREADS).flatten(order='F')
            out[i] = full[atoms.active]
    return out.T


def Jdag(atoms, inp):
    '''Conjugated forward transformation from real-space to reciprocal space.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

    Returns:
        Result as an array.
    '''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, atoms.S, order='F')
        out = ifftn(tmp, workers=THREADS).flatten(order='F')
    else:
        out = np.empty_like(inp, dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], atoms.S, order='F')
            out[i] = ifftn(tmp, workers=THREADS).flatten(order='F')
    return out.T


def T(atoms, inp, dr):
    '''Translation operator. Shifts input by the vector dr.

    Args:
        atoms :
            Atoms object.

        inp : array
            Coefficents input array.

        dr : array
            Real-space position vector.

    Returns:
        Result as an array.
    '''
    out = np.empty_like(inp, dtype=complex)
    factor = np.exp(-1j * np.dot(atoms.Gc, dr))
    for i in range(atoms.Ns):
        out[:, i] = factor * inp[:, i]
    return out
