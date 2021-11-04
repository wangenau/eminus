#!/usr/bin/env python3
'''
Calculate different energy contributions.
'''
import numpy as np
from numpy.linalg import inv
from scipy.special import erfc

from .exc import get_exc


class Energy:
    '''Energy class to save energy contributions in one place.'''
    __slots__ = ['Ekin', 'Eloc', 'Enonloc', 'Ecoul', 'Exc', 'Eewald', 'Esic']

    def __init__(self):
        self.Ekin = 0
        self.Eloc = 0
        self.Enonloc = 0
        self.Ecoul = 0
        self.Exc = 0
        self.Eewald = 0
        self.Esic = 0

    @property
    def Etot(self):
        '''Total energy is the sum of all energy contributions.'''
        return self.Ekin + self.Eloc + self.Enonloc + self.Ecoul + self.Exc + self.Eewald + \
               self.Esic

    def __repr__(self):
        out = ''
        for ie in self.__slots__:
            energy = eval('self.' + ie)
            if energy != 0:
                out = f'{out}{ie.ljust(8)}: {energy:+.9f} Eh\n'
        out = f'{out}{"-" * 25}\nEtot    : {self.Etot:+.9f} Eh'
        return out


def get_Ekin(atoms, Y):
    '''Calculate the kinetic energy.

    Args:
        atoms :
            Atoms object.

        Y : array
            Expansion coefficients of orthogornal wave functions.

    Returns:
        Kinetic energy in Hartree as a float.
    '''
    # Arias: Ekin = -0.5 Tr(F Cdag L(C))
    return np.real(-0.5 * np.trace(np.diag(atoms.f) @ (Y.conj().T @ atoms.L(Y))))


def get_Ecoul(atoms, n):
    '''Calculate the coulomb energy.

    Args:
        atoms :
            Atoms object.

        n : array
            Real-space electronic density.

    Returns:
        Coulomb energy in Hartree as a float.
    '''
    # Arias: Ecoul = -(Jn)dag O(phi)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    if atoms.cutcoul is None:
        return np.real(0.5 * n.conj().T @ atoms.Jdag(atoms.O(phi)))
    else:
        Rc = atoms.cutcoul
        correction = np.cos(np.sqrt(atoms.G2) * Rc) * atoms.O(phi)
        return np.real(0.5 * n.conj().T @ atoms.Jdag(atoms.O(phi) - correction))


def get_Exc(atoms, n, spinpol=False):
    '''Calculate the exchange-correlation energy.

    Args:
        atoms :
            Atoms object.

        n : array
            Real-space electronic density.

    Kwargs:
        spinpol : bool
            Choose if a spin-polarized exchange-correlation functional will be used.

    Returns:
        Exchange-correlation energy in Hartree as a float.
    '''
    # Arias: Exc = (Jn)dag O(J(exc))
    if atoms.spinpol or spinpol:
        exc = get_exc(atoms.exc, n, spinpol=True)[0]
    else:
        exc = get_exc(atoms.exc, n, spinpol=False)[0]
    return np.real(n.conj().T @ atoms.Jdag(atoms.O(atoms.J(exc))))


def get_Eloc(atoms, n):
    '''Calculate the local energy.

    Args:
        atoms :
            Atoms object.

        n : array
            Real-space electronic density.

    Returns:
        Local energy in Hartree as a float.
    '''
    return np.real(atoms.Vloc.conj().T @ n)


def get_Esic(atoms, n):
    '''Calculate the Perdew-Zunger self-interaction energy.'''
    # E_PZ-SIC = \sum_i Ecoul[n_i] + Exc[n_i, 0]
    Esic = 0
    for i in range(len(n)):
        # Normalize single-particle densities to 1
        n_tmp = n[i] / atoms.f[i]
        coul = get_Ecoul(atoms, n_tmp)
        # The exchange part for a SIC correction has to be spin polarized
        xc = get_Exc(atoms, n_tmp, spinpol=True)
        # SIC energy is scaled by the occupation
        Esic += (coul + xc) * atoms.f[i]
    return Esic


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_energies.jl
def get_Enonloc(atoms, Y):
    '''Calculate the non-local energy.

    Args:
        atoms :
            Atoms object.

        Y : array
            Expansion coefficients of orthogornal wave functions.

    Returns:
        Non-local energy in Hartree as a float.
    '''
    Enonloc = 0
    if atoms.NbetaNL > 0:  # Only calculate non-local potential if necessary
        Natoms = atoms.Natoms
        Nstates = atoms.Ns
        prj2beta = atoms.prj2beta
        betaNL = atoms.betaNL

        betaNL_psi = np.dot(Y.T.conj(), betaNL).conj()

        for ist in range(Nstates):
            enl = 0
            for ia in range(Natoms):
                psp = atoms.GTH[atoms.atom[ia]]
                for l in range(psp['lmax']):
                    for m in range(-l, l + 1):
                        for iprj in range(psp['Nproj_l'][l]):
                            ibeta = prj2beta[iprj, ia, l, m + psp['lmax'] - 1] - 1
                            for jprj in range(psp['Nproj_l'][l]):
                                jbeta = prj2beta[jprj, ia, l, m + psp['lmax'] - 1] - 1
                                hij = psp['h'][l, iprj, jprj]
                                enl += hij * np.real(betaNL_psi[ist, ibeta].conj() *
                                       betaNL_psi[ist, jbeta])
            Enonloc += atoms.f[ist] * enl
    # We have to multiply with the cell volume, because of different orthogonalization
    return Enonloc * atoms.CellVol


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_E_NN.jl
def get_Eewald(atoms, gcut=2, ebsl=1e-8):
    '''Calculate the Ewald energy.

    Args:
        atoms :
            Atoms object.

    Kwargs:
        gcut : float
            G-vector cut-off.

        ebsl : float
            Error tolerance

    Returns:
        Ewald energy in Hartree as a float.
    '''
    # For a plane wave code we have multiple contributions for the Ewald energy
    # Namely, a sum from contributions from real space, reciprocal space,
    # the self energy, (the dipole term [neglected]), and an additional electroneutrality term
    # See Eq. (4) https://juser.fz-juelich.de/record/16155/files/IAS_Series_06.pdf
    if atoms.cutcoul is not None:
        return 0

    Natoms = atoms.Natoms
    tau = atoms.X
    Zvals = atoms.Z
    omega = atoms.CellVol

    LatVecs = atoms.R
    t1 = LatVecs[0]
    t2 = LatVecs[1]
    t3 = LatVecs[2]
    t1m = np.sqrt(np.dot(t1, t1))
    t2m = np.sqrt(np.dot(t2, t2))
    t3m = np.sqrt(np.dot(t3, t3))

    RecVecs = 2 * np.pi * inv(LatVecs.conj().T)
    g1 = RecVecs[0]
    g2 = RecVecs[1]
    g3 = RecVecs[2]
    g1m = np.sqrt(np.dot(g1, g1))
    g2m = np.sqrt(np.dot(g2, g2))
    g3m = np.sqrt(np.dot(g3, g3))

    x = np.sum(Zvals**2)
    totalcharge = np.sum(Zvals)

    gexp = -np.log(ebsl)
    nu = 0.5 * np.sqrt(gcut**2 / gexp)

    tmax = np.sqrt(0.5 * gexp) / nu
    mmm1 = int(np.rint(tmax / t1m + 1.5))
    mmm2 = int(np.rint(tmax / t2m + 1.5))
    mmm3 = int(np.rint(tmax / t3m + 1.5))

    # Start by calculaton the self-energy
    Eewald = -nu * x / np.sqrt(np.pi)
    # Add the electroneutrality-term (Eq. 11)
    Eewald += -np.pi * totalcharge**2 / (2 * omega * nu**2)

    dtau = np.empty(3)
    G = np.empty(3)
    T = np.empty(3)
    for ia in range(Natoms):
        for ja in range(Natoms):
            dtau = tau[ia] - tau[ja]
            ZiZj = Zvals[ia] * Zvals[ja]
            for i in range(-mmm1, mmm1 + 1):
                for j in range(-mmm2, mmm2 + 1):
                    for k in range(-mmm3, mmm3 + 1):
                        if (ia != ja) or ((abs(i) + abs(j) + abs(k)) != 0):
                            T = i * t1 + j * t2 + k * t3
                            rmag = np.sqrt(np.sum((dtau - T)**2))
                            # Add the real space contribution
                            Eewald += 0.5 * ZiZj * erfc(rmag * nu) / rmag

    mmm1 = int(np.rint(gcut / g1m + 1.5))
    mmm2 = int(np.rint(gcut / g2m + 1.5))
    mmm3 = int(np.rint(gcut / g3m + 1.5))

    for ia in range(Natoms):
        for ja in range(Natoms):
            dtau = tau[ia] - tau[ja]
            ZiZj = Zvals[ia] * Zvals[ja]
            for i in range(-mmm1, mmm1 + 1):
                for j in range(-mmm2, mmm2 + 1):
                    for k in range(-mmm3, mmm3 + 1):
                        if (abs(i) + abs(j) + abs(k)) != 0:
                            G = i * g1 + j * g2 + k * g3
                            Gtau = np.sum(G * dtau)
                            G2 = np.sum(G**2)
                            # Add the reciprocal space contribution
                            x = 2 * np.pi / omega * np.exp(-0.25 * G2 / nu**2) / G2
                            Eewald += x * ZiZj * np.cos(Gtau)

    return Eewald
