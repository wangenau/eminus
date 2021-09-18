#!/usr/bin/env python3
'''
Utilities to localize and analyze orbitals.
'''
import numpy as np

from .scf import orth


def eval_psi(atoms, psi, r):
    '''Get the value for given psi a the coordinate point r.'''
    # Shift the evaluation point to (0,0,0), because we always have a lattice point there
    psi_T = atoms.T(psi, -r)
    psi_Trs = atoms.I(psi_T)
    # The zero entry is always the value at point (0,0,0)
    return psi_Trs[0]


def get_R(atoms, psi, fods):
    '''Calculate the transformation matrix R to generate FOs.'''
    # We only calculate occupied orbitals, so a zero matrix is enough
    R = np.zeros((atoms.Ns, atoms.Ns), dtype=complex)
    for i in range(len(fods)):
        # Get the value at one FOD position for all psi
        psi_fod = eval_psi(atoms, psi, fods[i])
        sum_psi_fod = np.sqrt(np.sum(psi_fod.conj() * psi_fod))
        for j in range(len(fods)):
            R[i, j] = psi_fod[j].conj() / sum_psi_fod
    return R


def get_FOs(atoms, psi, fods):
    '''Calculate FOs given a psi in reciprocal space and a set of FODs.'''
    FOs = np.zeros((len(atoms.r), atoms.Ns), dtype=complex)
    # Get the transformation matrix R
    R = get_R(atoms, psi, fods)
    # Transform psi to real space
    psi_rs = atoms.I(psi)
    for i in range(len(R)):
        for j in range(atoms.Ns):
            FOs[:, i] += R[i, j] * psi_rs[:, j]
    return FOs


def get_FLOs(atoms, psi, fods):
    '''Calculate FLOs for given FOs by orthogornalizing them.'''
    FOs = get_FOs(atoms, psi, fods)
    return orth(atoms, FOs)
