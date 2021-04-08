#!/usr/bin/env python3
'''
Calculate the non-local potential with GTH pseudopotentials. Phys. Rev. B 54, 1703
'''
import numpy as np
from numpy.linalg import norm
from .read_gth import read_GTH
from .utils import Ylm_real


# Adopted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_energies.jl
def calc_Enl(a, W):
    '''Calculate the non-local energy.'''
    # Parameters of the non-local pseudopotential
    NbetaNL = a.NbetaNL
    prj2beta = a.prj2beta
    betaNL = a.betaNL

    Natoms = len(a.atom)
    Nstates = a.Ns

    betaNL_psi = np.dot(W.T.conj(), betaNL).conj()

    E_Ps_nloc = 0
    for ist in range(Nstates):
        enl1 = 0
        for ia in range(Natoms):
            psp = a.GTH[a.atom[ia]]
            for l in range(psp['lmax']):
                for m in range(-l, l + 1):
                    for iprj in range(psp['Nproj_l'][l]):
                        ibeta = prj2beta[iprj, ia, l, m + psp['lmax'] - 1] - 1
                        for jprj in range(psp['Nproj_l'][l]):
                            jbeta = prj2beta[jprj, ia, l, m + psp['lmax'] - 1] - 1
                            hij = psp['h'][l, iprj, jprj]
                            enl1 += hij * np.real(betaNL_psi[ist, ibeta].conj() * betaNL_psi[ist, jbeta])
        E_Ps_nloc += a.f[ist] * enl1
    return E_Ps_nloc


# Adopted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/op_V_Ps_nloc.jl
def calc_Vnl(a, W):
    '''Calculate the non-local pseudopotential, applied on the basis functions W.'''
    # Parameters of the non-local pseudopotential
    NbetaNL = a.NbetaNL
    prj2beta = a.prj2beta
    betaNL = a.betaNL

    Natoms = len(a.atom)
    Npoints = len(W)
    Nstates = a.Ns

    betaNL_psi = np.dot(W.T.conj(), betaNL).conj()

    Vpsi = np.zeros([Npoints, Nstates], dtype=complex)
    for ist in range(Nstates):
        for ia in range(Natoms):
            psp = a.GTH[a.atom[ia]]
            for l in range(psp['lmax']):
                for m in range(-l, l + 1):
                    for iprj in range(psp['Nproj_l'][l]):
                        ibeta = prj2beta[iprj, ia, l, m + psp['lmax'] - 1] - 1
                        for jprj in range(psp['Nproj_l'][l]):
                            jbeta = prj2beta[jprj, ia, l, m + psp['lmax'] - 1] - 1
                            hij = psp['h'][l, iprj, jprj]
                            Vpsi[:, ist] += hij * betaNL[:, ibeta] * betaNL_psi[ist, jbeta]
    return Vpsi


# Adopted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/PsPotNL.jl
def init_gth_nonloc(a):
    '''Initialize parameters to calculate non-local contributions.'''
    Natoms = len(a.atom)
    Npoints = len(a.active[0])
    CellVol = a.CellVol

    prj2beta = np.zeros([3, Natoms, 4, 7], dtype=int)
    prj2beta[:] = -1  # Set to invalid index

    NbetaNL = 0
    for ia in range(Natoms):
        psp = a.GTH[a.atom[ia]]
        for l in range(psp['lmax']):
            for m in range(-l, l + 1):
                for iprj in range(psp['Nproj_l'][l]):
                    NbetaNL += 1
                    prj2beta[iprj, ia, l, m + psp['lmax'] - 1] = NbetaNL

    # Sort G-vectors by their magnitude
    # PWDFT.jl uses sortperm, for compareabilty we need to sort with mergesort
    idx = np.argsort(a.G2c, kind='mergesort')

    # Can be calculated outside the loop in this case
    g = a.Gc[idx]  # Simplified, would normally be G+k
    Gm = np.sqrt(a.G2c[idx])

    ibeta = 0
    betaNL = np.zeros([Npoints, NbetaNL], dtype=complex)
    for ia in range(Natoms):
        psp = a.GTH[a.atom[ia]]
        for l in range(psp['lmax']):
            for m in range(-l, l + 1):
                for iprj in range(psp['Nproj_l'][l]):
                    GX = np.sum(a.X[ia] * g, axis=1)
                    Sf = np.cos(GX) - 1j * np.sin(GX)
                    betaNL[:, ibeta] = (-1j)**l * Ylm_real(l, m, g) * eval_proj_G(psp, l, iprj + 1, Gm, CellVol) * Sf
                    ibeta += 1
    return NbetaNL, prj2beta, betaNL


# Adopted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/PsPot_GTH.jl
def eval_proj_G(psp, l, iproj, Gm, CellVol):
    '''Evaluate GTH projector function in G-space.'''
    rrl = psp['rc'][l]
    Gr2 = (Gm * rrl)**2

    if l == 0:  # s-channel
        if iproj == 1:
            Vprj = np.exp(-0.5 * Gr2)
        elif iproj == 2:
            Vprj = 2 / np.sqrt(15) * np.exp(-0.5 * Gr2) * (3 - Gr2)
        elif iproj == 3:
            Vprj = (4 / 3) / np.sqrt(105) * np.exp(-0.5 * Gr2) * (15 - 10 * Gr2 + Gr2**2)
    elif l == 1:  # p-channel
        if iproj == 1:
            Vprj = (1 / np.sqrt(3)) * np.exp(-0.5 * Gr2) * Gm
        elif iproj == 2:
            Vprj = (2 / np.sqrt(105)) * np.exp(-0.5 * Gr2) * Gm * (5 - Gr2)
        elif iproj == 3:
            Vprj = (4 / 3) / np.sqrt(1155) * np.exp(-0.5 * Gr2) * Gm * (35 - 14 * Gr2 + Gr2**2)
    elif l == 2:  # d-channel
        if iproj == 1:
            Vprj = (1 / np.sqrt(15)) * np.exp(-0.5 * Gr2) * Gm**2
        elif iproj == 2:
            Vprj = (2 / 3) / np.sqrt(105) * np.exp(-0.5 * Gr2) * Gm**2 * (7 - Gr2)
    elif l == 3:  # f-channel
        # Only one projector
        Vprj = Gm**3 * np.exp(-0.5 * Gr2) / np.sqrt(105)
    else:
        print(f'ERROR: No projector found for l={l}')

    pre = 4 * np.pi**(5 / 4) * np.sqrt(2**(l + 1) * rrl**(2 * l + 3) / CellVol)
    return pre * Vprj
