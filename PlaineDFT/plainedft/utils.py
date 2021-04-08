#!/usr/bin/env python3
'''
Linear algebra calculation utilities.
'''
import numpy as np
from numpy.linalg import norm


# FIXME: This functions works, but is unused.
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


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/Ylm_real.jl
def Ylm_real(l, m, R):
    '''Calculate spherical harmonics.'''
    if l == 0:
        return 0.5 * np.sqrt(1 / np.pi) * np.ones(len(R))

    eps = 1e-9
    Rmod = norm(R, axis=1)

    cost = np.zeros(len(R))
    cost_idx = Rmod > eps
    cost[cost_idx] = R[cost_idx, 2] / Rmod[cost_idx]

    sint = np.sqrt(np.amax([np.zeros(len(R)), 1 - cost**2], axis=0))

    phi = -np.pi / 2 * np.ones(len(R))
    phi_idx = R[:, 1] >= 0
    phi[phi_idx] = np.pi / 2 * np.ones(len(phi[phi_idx]))

    phi_idx = R[:, 0] < -eps
    phi[phi_idx] = np.arctan(R[phi_idx, 1] / R[phi_idx, 0]) + np.pi

    phi_idx = R[:, 0] > eps
    phi[phi_idx] = np.arctan(R[phi_idx, 1] / R[phi_idx, 0])

    if l == 1:
        # py
        if m == -1:
            ylm = 0.5 * np.sqrt(3 / np.pi) * sint * np.sin(phi)
        # pz
        elif m == 0:
            ylm = 0.5 * np.sqrt(3 / np.pi) * cost
        # px
        elif m == 1:
            ylm = 0.5 * np.sqrt(3 / np.pi) * sint * np.cos(phi)
        else:
            print(f'ERROR: No definition found for Ylm({l}, {m})')
    elif l == 2:
        # dxy
        if m == -2:
            ylm = np.sqrt(15 / 16 / np.pi) * sint**2 * np.sin(2 * phi)
        # dyz
        elif m == -1:
            ylm = np.sqrt(15 / 4 / np.pi) * cost * sint * np.sin(phi)
        # dz2
        elif m == 0:
            ylm = 0.25 * np.sqrt(5 / np.pi) * (3 * cost**2 - 1)
        # dxz
        elif m == 1:
            ylm = np.sqrt(15 / 4 / np.pi) * cost * sint * np.cos(phi)
        # dx2-y2
        elif m == 2:
            ylm = np.sqrt(15 / 16 / np.pi) * sint**2 * np.cos(2 * phi)
        else:
            print(f'ERROR: No definition found for Ylm({l}, {m})')
    elif l == 3:
        if m == -3:
            ylm = 0.25 * np.sqrt(35 / 2 / np.pi) * sint**3 * np.sin(3 * phi)
        elif m == -2:
            ylm = 0.25 * np.sqrt(105 / np.pi) * sint**2 * cost * np.sin(2 * phi)
        elif m == -1:
            ylm = 0.25 * np.sqrt(21 / 2 / np.pi) * sint * (5 * cost**2 - 1) * np.sin(phi)
        elif m == 0:
            ylm = 0.25 * np.sqrt(7 / np.pi) * (5 * cost**3 - 3 * cost)
        elif m == 1:
            ylm = 0.25 * np.sqrt(21 / 2 / np.pi) * sint * (5 * cost**2 - 1) * np.cos(phi)
        elif m == 2:
            ylm = 0.25 * np.sqrt(105 / np.pi) * sint**2 * cost * np.cos(2 * phi)
        elif m == 3:
            ylm = 0.25 * np.sqrt(35 / 2 / np.pi) * sint**3 * np.cos(3 * phi)
        else:
            print(f'ERROR: No definition found for Ylm({l}, {m})')
    else:
        print(f'ERROR: No definition found for Ylm({l}, {m})')
    return ylm


def eval_proj_G(psp, l, iproj, Gm, CellVol):
    rrl = psp['rc'][l]
    Gr2 = (Gm * rrl)**2

    # s-channel
    if l == 0:
        if iproj == 1:
            Vprj = np.exp(-0.5 * Gr2)
        elif iproj == 2:
            Vprj = 2 / np.sqrt(15) * np.exp(-0.5 * Gr2) * (3 - Gr2)
        elif iproj == 3:
            Vprj = (4 / 3) / np.sqrt(105) * np.exp(-0.5 * Gr2) * (15 - 10 * Gr2 + Gr2**2)
    # p-channel
    elif l == 1:
        if iproj == 1:
            Vprj = (1 / np.sqrt(3)) * np.exp(-0.5 * Gr2) * Gm
        elif iproj == 2:
            Vprj = (2 / np.sqrt(105)) * np.exp(-0.5 * Gr2) * Gm * (5 - Gr2)
        elif iproj == 3:
            Vprj = (4 / 3) / np.sqrt(1155) * np.exp(-0.5 * Gr2) * Gm * (35 - 14 * Gr2 + Gr2**2)
    # d-channel
    elif l == 2:
        if iproj == 1:
            Vprj = (1 / np.sqrt(15)) * np.exp(-0.5 * Gr2) * Gm**2
        elif iproj == 2:
            Vprj = (2 / 3) / np.sqrt(105) * np.exp(-0.5 * Gr2) * Gm**2 * (7 - Gr2)
    # f-channel
    elif l == 3:
        # Only one projector
        Vprj = Gm**3 * np.exp(-0.5 * Gr2) / np.sqrt(105)
    else:
        print(f'ERROR: No projector found for l={l}')

    pre = 4 * np.pi**(5 / 4) * np.sqrt(2**(l + 1) * rrl**(2 * l + 3) / CellVol)
    return pre * Vprj
