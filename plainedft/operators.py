#!/usr/bin/env python3
'''
Basis set depedent operators for DFT calculations in plane-wave basis.
'''
import numpy as np
from numpy.linalg import det
from scipy.fft import ifftn, fftn
import os

THREADS = int(os.environ['OMP_NUM_THREADS'])


def O(atoms, inp):
    '''Overlap operator.'''
    return det(atoms.R) * inp


def L(atoms, inp):
    '''Laplacian operator.'''
    inp = inp.T
    if inp.shape[1] == len(atoms.G2c):
        return (-det(atoms.R) * atoms.G2c * inp).T
    else:
        return (-det(atoms.R) * atoms.G2 * inp).T


def Linv(atoms, inp):
    '''Inverse Laplacian operator.'''
    inp = inp.T
    out = np.zeros(inp.shape, dtype=complex)
    if inp.ndim == 1:
        out[1:] = inp[1:] / atoms.G2[1:] / -det(atoms.R)
    else:
        for i in range(len(inp)):
            out[i][1:] = inp[i][1:] / atoms.G2[1:] / -det(atoms.R)
    return out.T


def K(atoms, inp):
    '''Precondition by applying 1/(1+G2) to the input.'''
    inp = inp.T
    out = np.empty(inp.shape, dtype=complex)
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
    '''Forward transformation from real-space to reciprocal space.'''
    inp = inp.T
    if inp.ndim == 1:
        inp = np.array([inp])
    if np.size(inp, 1) == np.prod(atoms.S):
        out = np.empty(inp.shape, dtype=complex)
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


def J(atoms, inp):
    '''Backwards transformation from reciprocal space to real-space.'''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, atoms.S, order='F')
        out = fftn(tmp, workers=THREADS).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(inp.shape[0]):
            tmp = np.reshape(inp[i], atoms.S, order='F')
            out[i] = fftn(tmp, workers=THREADS).flatten(order='F')
    return (out / np.prod(atoms.S)).T


def Idag(atoms, inp):
    '''Conjugated forward transformation from real-space to reciprocal space.'''
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
    '''Conjugated backwards transformation from reciprocal space to real-space.'''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, atoms.S, order='F')
        out = ifftn(tmp, workers=THREADS).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], atoms.S, order='F')
            out[i] = ifftn(tmp, workers=THREADS).flatten(order='F')
    return out.T
