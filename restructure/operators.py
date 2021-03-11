#!/usr/bin/env python3
'''
Basis set depedent operators for DFT calculations in plane-wave basis.
'''
import numpy as np
from numpy.linalg import det
from numpy.fft import ifftn, fftn


def O(a, inp):
    '''Overlap operator.'''
    return det(a.R) * inp


def L(a, inp):
    '''Laplacian operator.'''
    inp = inp.T
    if inp.shape[1] == len(a.G2c):
        return (-det(a.R) * a.G2c * inp).T
    else:
        return (-det(a.R) * a.G2 * inp).T


def Linv(a, inp):
    '''Inverse Laplacian operator.'''
    inp = inp.T
    out = np.zeros(inp.shape, dtype=complex)
    if inp.ndim == 1:
        out[1:] = inp[1:] / a.G2[1:] / -det(a.R)
    else:
        for i in range(len(inp)):
            out[i][1:] = inp[i][1:] / a.G2[1:] / -det(a.R)
    return out.T


def K(a, inp):
    '''Precondition by applying 1/(1+G2) to the input.'''
    inp = inp.T
    out = np.empty(inp.shape, dtype=complex)
    if inp.shape[1] == len(a.G2c):
        if inp.ndim == 1:
            out = inp / (1 + a.G2c)
        else:
            for i in range(len(inp)):
                out[i] = inp[i] / (1 + a.G2c)
    else:
        if inp.ndim == 1:
            out = inp / (1 + a.G2)
        else:
            for i in range(len(inp)):
                out[i] = inp[i] / (1 + a.G2)
    return out.T


def I(a, inp):
    '''Forward transformation from real-space to reciprocal space.'''
    inp = inp.T
    if inp.ndim == 1:
        inp = np.array([inp])
    if np.size(inp, 1) == np.prod(a.S):
        out = np.empty(inp.shape, dtype=complex)
        for i in range(inp.shape[0]):
            tmp = np.reshape(inp[i], a.S, order='F')
            out[i] = ifftn(tmp).flatten(order='F')
    else:
        out = np.empty((inp.shape[0], np.prod(a.S)), dtype=complex)
        for i in range(inp.shape[0]):
            full = np.zeros(np.prod(a.S), dtype=complex)
            full[a.active] = inp[i]
            full = np.reshape(full, a.S, order='F')
            out[i] = ifftn(full).flatten(order='F')
    return (out * np.prod(a.S)).T


def J(a, inp):
    '''Backwards transformation from reciprocal space to real-space.'''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, a.S, order='F')
        out = fftn(tmp).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(inp.shape[0]):
            tmp = np.reshape(inp[i], a.S, order='F')
            out[i] = fftn(tmp).flatten(order='F')
    return (out / np.prod(a.S)).T


def Idag(a, inp):
    '''Conjugated forward transformation from real-space to reciprocal space.'''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, a.S, order='F')
        full = fftn(tmp).flatten(order='F')
        out = full[a.active]
    else:
        out = np.empty((np.size(inp, 0), len(a.active[0])), dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], a.S, order='F')
            full = fftn(tmp).flatten(order='F')
            out[i] = full[a.active]
    return out.T


def Jdag(a, inp):
    '''Conjugated backwards transformation from reciprocal space to real-space.'''
    inp = inp.T
    if inp.ndim == 1:
        tmp = np.reshape(inp, a.S, order='F')
        out = ifftn(tmp).flatten(order='F')
    else:
        out = np.empty(inp.shape, dtype=complex)
        for i in range(len(inp)):
            tmp = np.reshape(inp[i], a.S, order='F')
            out[i] = ifftn(tmp).flatten(order='F')
    return out.T
