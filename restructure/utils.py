#!/usr/bin/env python3
import numpy as np


# def diagouter(A, B):
#     '''Calculate the expression Diag (A * Bdag).'''
#     return np.sum(A * B.conj(), axis=1)


def Diagprod(a, B):
    '''Calculate the expression Diag(a) * B.'''
    B = B.T
    return (a * B).T


def dotprod(a, b):
    '''Calculate the expression a * b.'''
    return np.real(np.trace(a.conj().T @ b))
