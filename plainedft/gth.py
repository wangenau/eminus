#!/usr/bin/env python3
'''
Utilities to use Goedecker, Teter, and Hutter pseudopotentials.
'''
from glob import glob
from os.path import basename

import numpy as np

from . import __path__
from .utils import Ylm_real


def init_gth_loc(atoms):
    '''Initialize parameters to calculate local contributions of GTH pseudopotentials.

    Args:
        atoms :
            Atoms object.

    Returns:
        Local GTH potential contribution as an array.
    '''
    G2 = atoms.G2
    atom = atoms.atom
    species = set(atom)

    Vsp = np.empty(len(G2))  # Potential for every species
    Vloc = np.zeros(len(G2))  # Total local potential
    for isp in species:
        psp = atoms.GTH[isp]
        rloc = psp['rlocal']
        Zion = psp['Zval']
        C1 = psp['C'][0]
        C2 = psp['C'][1]
        C3 = psp['C'][2]
        C4 = psp['C'][3]

        omega = 1  # Normally this would be det(atoms.R), but Arias notation is off by this factor
        rlocG2 = G2[1:] * rloc**2

        Vsp[1:] = -4 * np.pi * Zion / omega * np.exp(-0.5 * rlocG2) / G2[1:] + \
                  np.sqrt((2 * np.pi)**3) * rloc**3 / omega * np.exp(-0.5 * rlocG2) * \
                  (C1 + C2 * (3 - rlocG2) + C3 * (15 - 10 * rlocG2 + rlocG2**2) +
                  C4 * (105 - 105 * rlocG2 + 21 * rlocG2**2 - rlocG2**3))
        Vsp[0] = 2 * np.pi * Zion * rloc**2 + \
                 (2 * np.pi)**1.5 * rloc**3 * (C1 + 3 * C2 + 15 * C3 + 105 * C4)

        # Sum up the structure factor for every species
        Sf = np.zeros(len(atoms.Sf[0]), dtype=complex)
        for ia in range(len(atom)):
            if atom[ia] == isp:
                Sf += atoms.Sf[ia]
        Vloc += np.real(atoms.J(Vsp * Sf))
    return Vloc


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/PsPotNL.jl
def init_gth_nonloc(atoms):
    '''Initialize parameters to calculate non-local contributions of GTH pseudopotentials.

    Args:
        atoms :
            Atoms object.

    Returns:
        NbetaNL, prj2beta, and betaNL as a tuple(int, array, array).
    '''
    Natoms = atoms.Natoms
    Npoints = len(atoms.active[0])
    CellVol = atoms.CellVol

    prj2beta = np.zeros([3, Natoms, 4, 7], dtype=int)
    prj2beta[:] = -1  # Set to invalid index

    NbetaNL = 0
    for ia in range(Natoms):
        psp = atoms.GTH[atoms.atom[ia]]
        for l in range(psp['lmax']):
            for m in range(-l, l + 1):
                for iprj in range(psp['Nproj_l'][l]):
                    NbetaNL += 1
                    prj2beta[iprj, ia, l, m + psp['lmax'] - 1] = NbetaNL

    g = atoms.Gc  # Simplified, would normally be G+k
    Gm = np.sqrt(atoms.G2c)

    ibeta = 0
    betaNL = np.zeros([Npoints, NbetaNL], dtype=complex)
    for ia in range(Natoms):
        Sf = atoms.Idag(atoms.J(atoms.Sf[ia]))
        psp = atoms.GTH[atoms.atom[ia]]
        for l in range(psp['lmax']):
            for m in range(-l, l + 1):
                for iprj in range(psp['Nproj_l'][l]):
                    betaNL[:, ibeta] = (-1j)**l * Ylm_real(l, m, g) * \
                                       eval_proj_G(psp, l, iprj + 1, Gm, CellVol) * Sf
                    ibeta += 1
    return NbetaNL, prj2beta, betaNL


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/op_V_Ps_nloc.jl
def calc_Vnonloc(atoms, W):
    '''Calculate the non-local pseudopotential, applied on the basis functions W.

    Args:
        atoms :
            Atoms object.

        W : array
            Expansion coefficients of unconstrained wave functions.

    Returns:
        Non-local GTH potential contribution as an array.
    '''
    Npoints = len(W)
    Nstates = atoms.Ns

    Vpsi = np.zeros([Npoints, Nstates], dtype=complex)
    if atoms.NbetaNL > 0:  # Only calculate non-local potential if necessary
        Natoms = atoms.Natoms
        prj2beta = atoms.prj2beta
        betaNL = atoms.betaNL

        betaNL_psi = np.dot(W.T.conj(), betaNL).conj()

        for ist in range(Nstates):
            for ia in range(Natoms):
                psp = atoms.GTH[atoms.atom[ia]]
                for l in range(psp['lmax']):
                    for m in range(-l, l + 1):
                        for iprj in range(psp['Nproj_l'][l]):
                            ibeta = prj2beta[iprj, ia, l, m + psp['lmax'] - 1] - 1
                            for jprj in range(psp['Nproj_l'][l]):
                                jbeta = prj2beta[jprj, ia, l, m + psp['lmax'] - 1] - 1
                                hij = psp['h'][l, iprj, jprj]
                                Vpsi[:, ist] += hij * betaNL[:, ibeta] * betaNL_psi[ist, jbeta]
    # We have to multiply with the cell volume, because of different orthogonalization
    return Vpsi * atoms.CellVol


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/PsPot_GTH.jl
def eval_proj_G(psp, l, iprj, Gm, CellVol):
    '''Evaluate GTH projector functions in G-space.

    Args:
        psp : dict
            GTH parameters.

        l : int
            Angular momentum number.

        iprj : int
            Nproj_l index.

        Gm : array
            Magnitude of G-vectors.

        CellVol : float
            Unit cell volume.

    Returns:
        GTH projector as an array.
    '''
    rrl = psp['rc'][l]
    Gr2 = (Gm * rrl)**2

    if l == 0:  # s-channel
        if iprj == 1:
            Vprj = np.exp(-0.5 * Gr2)
        elif iprj == 2:
            Vprj = 2 / np.sqrt(15) * np.exp(-0.5 * Gr2) * (3 - Gr2)
        elif iprj == 3:
            Vprj = (4 / 3) / np.sqrt(105) * np.exp(-0.5 * Gr2) * (15 - 10 * Gr2 + Gr2**2)
    elif l == 1:  # p-channel
        if iprj == 1:
            Vprj = (1 / np.sqrt(3)) * np.exp(-0.5 * Gr2) * Gm
        elif iprj == 2:
            Vprj = (2 / np.sqrt(105)) * np.exp(-0.5 * Gr2) * Gm * (5 - Gr2)
        elif iprj == 3:
            Vprj = (4 / 3) / np.sqrt(1155) * np.exp(-0.5 * Gr2) * Gm * (35 - 14 * Gr2 + Gr2**2)
    elif l == 2:  # d-channel
        if iprj == 1:
            Vprj = (1 / np.sqrt(15)) * np.exp(-0.5 * Gr2) * Gm**2
        elif iprj == 2:
            Vprj = (2 / 3) / np.sqrt(105) * np.exp(-0.5 * Gr2) * Gm**2 * (7 - Gr2)
    elif l == 3:  # f-channel
        # Only one projector
        Vprj = Gm**3 * np.exp(-0.5 * Gr2) / np.sqrt(105)
    else:
        print(f'ERROR: No projector found for l={l}')

    pre = 4 * np.pi**(5 / 4) * np.sqrt(2**(l + 1) * rrl**(2 * l + 3) / CellVol)
    return pre * Vprj


def read_gth(system, charge=None, psp_path=None):
    '''Read GTH files for a given system.

    Args:
        system : str
            Atom name.

    Kwargs:
        charge : int
            Valence charge.

        psp_path : str
            Path to GTH pseudopotential files. None will default to /installation_path/pade_gth/.

    Returns:
        GTH parameters as a dictionary.
    '''
    if psp_path is None:
        psp_path = f'{__path__[0]}/pade_gth/'

    if charge is not None:
        f_psp = f'{psp_path}{system}-q{charge}.gth'
    else:
        files = glob(f'{psp_path}{system}-q*')
        files.sort()
        try:
            f_psp = files[0]
        except IndexError:
            print(f'ERROR: There is no GTH pseudopotential in {psp_path} for "{system}"')
        if len(files) > 1:
            print(f'INFO: Multiple pseudopotentials found for "{system}". '
                  f'Continue with "{basename(f_psp)}".')

    psp = {}
    C = np.zeros(4)
    rc = np.zeros(4)
    Nproj_l = np.zeros(4, dtype=int)
    h = np.zeros([4, 3, 3])
    try:
        with open(f_psp, 'r') as fh:
            psp['symbol'] = fh.readline().split()[0]
            N_all = fh.readline().split()
            N_s, N_p, N_d, N_f = int(N_all[0]), int(N_all[1]), int(N_all[2]), int(N_all[3])
            psp['Zval'] = N_s + N_p + N_d + N_f
            rlocal, n_c_local = fh.readline().split()
            psp['rlocal'] = float(rlocal)
            psp['n_c_local'] = int(n_c_local)
            for i, val in enumerate(fh.readline().split()):
                C[i] = float(val)
            psp['C'] = C
            lmax = int(fh.readline().split()[0])
            psp['lmax'] = lmax
            for iiter in range(lmax):
                rc[iiter], Nproj_l[iiter] = [float(i) for i in fh.readline().split()]
                for jiter in range(Nproj_l[iiter]):
                    tmp = fh.readline().split()
                    for iread, kiter in enumerate(range(jiter, Nproj_l[iiter])):
                        h[iiter, jiter, kiter] = float(tmp[iread])
            psp['rc'] = rc
            psp['Nproj_l'] = Nproj_l
            for k in range(3):
                for i in range(2):
                    for j in range(i + 1, 2):
                        h[k, j, i] = h[k, i, j]
            psp['h'] = h
    except FileNotFoundError:
        print(f'ERROR: There is no GTH pseudopotential for "{basename(f_psp)}"')
    return psp
