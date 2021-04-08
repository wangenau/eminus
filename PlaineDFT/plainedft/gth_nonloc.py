#!/usr/bin/env python3
'''
Calculate the non-local potential with GTH pseudopotentials. Phys. Rev. B 54, 1703
'''
import numpy as np
from numpy.linalg import norm
from .read_gth import read_GTH
from .utils import Ylm_real, eval_proj_G


def calc_Enl(a, W):
    '''Calculate the non-local energy.'''
    # Parameters for the non-local pseudopotential
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


def calc_Vnl(a, W):
    '''Calculate the non-local pseudopotential, applied on the basis functions W.'''
    # Parameters for the non-local pseudopotential
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
                            for igk in range(Npoints):
                                Vpsi[igk, ist] += hij * betaNL[igk, ibeta] * betaNL_psi[ist, jbeta]
    return Vpsi


def init_gth_nonloc(a):
    '''Initialize parameters to calculate non-local contributions.'''
    Natoms = len(a.atom)
    Npoints = len(a.active[0])
    CellVol = a.a**3  # We only have cubic unit cells for now

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

    betaNL = np.zeros([Npoints, NbetaNL], dtype=complex)

    ibeta = 0
    for ia in range(Natoms):
        psp = a.GTH[a.atom[ia]]
        for l in range(psp['lmax']):
            for m in range(-l, l + 1):
                for iprj in range(psp['Nproj_l'][l]):
                    for igk in range(Npoints):
                        g = a.Gc[igk]  # Simplified, would normally be G+k
                        Gm = norm(g)
                        GX = np.sum(a.X[ia] * g)
                        Sf = np.cos(GX) - 1j * np.sin(GX)
                        betaNL[igk, ibeta] = (-1j)**l * Ylm_real(l, m, g) * eval_proj_G(psp, l, iprj + 1, Gm, CellVol) * Sf
                    ibeta += 1
    return NbetaNL, prj2beta, betaNL
