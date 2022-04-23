#!/usr/bin/env python3
'''Calculate different energy contributions.'''
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
        '''Printe the energies stored in the Energy object.'''
        out = ''
        for ie in self.__slots__:
            energy = eval('self.' + ie)
            if energy != 0:
                out = f'{out}{ie.ljust(8)}: {energy:+.9f} Eh\n'
        out = f'{out}{"-" * 25}\nEtot    : {self.Etot:+.9f} Eh'
        return out


def get_Ekin(atoms, Y):
    '''Calculate the kinetic energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y (array): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        float: Kinetic energy in Hartree.
    '''
    # Ekin = -0.5 Tr(F Wdag L(W))
    return np.real(-0.5 * np.trace(np.diag(atoms.f) @ (Y.conj().T @ atoms.L(Y))))


def get_Ecoul(atoms, n):
    '''Calculate the Coulomb energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n (array): Real-space electronic density.

    Returns:
        float: Coulomb energy in Hartree.
    '''
    # Ecoul = -(J(n))dag O(phi)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    return np.real(0.5 * n.conj().T @ atoms.Jdag(atoms.O(phi)))


def get_Exc(atoms, n, spinpol=False):
    '''Calculate the exchange-correlation energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n (array): Real-space electronic density.

    Keyword Args:
        spinpol (bool): Choose if a spin-polarized exchange-correlation functional will be used.

    Returns:
        float: Exchange-correlation energy in Hartree.
    '''
    # Exc = (J(n))dag O(J(exc))
    exc = get_exc(atoms.exc, n, 'density', spinpol)
    return np.real(n.conj().T @ atoms.Jdag(atoms.O(atoms.J(exc))))


def get_Eloc(atoms, n):
    '''Calculate the local energy contribution.

    Args:
        atoms: Atoms object.
        n (array): Real-space electronic density.

    Returns:
        float: Local energy contribution in Hartree.
    '''
    return np.real(atoms.Vloc.conj().T @ n)


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_energies.jl
def get_Enonloc(atoms, Y):
    '''Calculate the non-local GTH energy contribution.

    Reference: Phys. Rev. B 54, 1703.

    Args:
        atoms: Atoms object.
        Y (array): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        float: Non-local GTH energy contribution in Hartree.
    '''
    Enonloc = 0
    if atoms.NbetaNL > 0:  # Only calculate non-local potential if necessary
        Natoms = atoms.Natoms
        Nstates = atoms.Ns
        prj2beta = atoms.prj2beta
        betaNL = atoms.betaNL

        betaNL_psi = (Y.conj().T @ betaNL).conj()

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
    # We have to multiply with the cell volume, because of different orthogonalization methods
    return Enonloc * atoms.Omega


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_E_NN.jl
def get_Eewald(atoms, gcut=2, gamma=1e-8):
    '''Calculate the Ewald energy.

    Reference: J. Chem. Theory Comput. 10, 381.

    Args:
        atoms: Atoms object.

    Keyword Args:
        gcut (float): G-vector cut-off.
        gamma (float): Error tolerance

    Returns:
        float: Ewald energy in Hartree.
    '''
    # For a plane wave code we have multiple contributions for the Ewald energy
    # Namely, a sum from contributions from real-space, reciprocal space,
    # the self energy, (the dipole term [neglected]), and an additional electroneutrality term
    Natoms = atoms.Natoms
    X = atoms.X
    Z = atoms.Z
    Omega = atoms.Omega
    R = atoms.R

    t1, t2, t3 = R
    t1m = np.sqrt(t1 @ t1)
    t2m = np.sqrt(t2 @ t2)
    t3m = np.sqrt(t3 @ t3)

    g1, g2, g3 = 2 * np.pi * inv(R.conj().T)
    g1m = np.sqrt(g1 @ g1)
    g2m = np.sqrt(g2 @ g2)
    g3m = np.sqrt(g3 @ g3)

    gexp = -np.log(gamma)
    nu = 0.5 * np.sqrt(gcut**2 / gexp)

    x = np.sum(Z**2)
    totalcharge = np.sum(Z)

    # Start by calculaton the self-energy
    Eewald = -nu * x / np.sqrt(np.pi)
    # Add the electroneutrality-term
    Eewald += -np.pi * totalcharge**2 / (2 * Omega * nu**2)

    tmax = np.sqrt(0.5 * gexp) / nu
    mmm1 = np.rint(tmax / t1m + 1.5)
    mmm2 = np.rint(tmax / t2m + 1.5)
    mmm3 = np.rint(tmax / t3m + 1.5)

    dX = np.empty(3)
    T = np.empty(3)
    for ia in range(Natoms):
        for ja in range(Natoms):
            dX = X[ia] - X[ja]
            ZiZj = Z[ia] * Z[ja]
            for i in np.arange(-mmm1, mmm1 + 1):
                for j in np.arange(-mmm2, mmm2 + 1):
                    for k in np.arange(-mmm3, mmm3 + 1):
                        if (ia != ja) or ((abs(i) + abs(j) + abs(k)) != 0):
                            T = i * t1 + j * t2 + k * t3
                            rmag = np.sqrt(np.sum((dX - T)**2))
                            # Add the real-space contribution
                            Eewald += 0.5 * ZiZj * erfc(rmag * nu) / rmag

    mmm1 = np.rint(gcut / g1m + 1.5)
    mmm2 = np.rint(gcut / g2m + 1.5)
    mmm3 = np.rint(gcut / g3m + 1.5)

    G = np.empty(3)
    for ia in range(Natoms):
        for ja in range(Natoms):
            dX = X[ia] - X[ja]
            ZiZj = Z[ia] * Z[ja]
            for i in np.arange(-mmm1, mmm1 + 1):
                for j in np.arange(-mmm2, mmm2 + 1):
                    for k in np.arange(-mmm3, mmm3 + 1):
                        if (abs(i) + abs(j) + abs(k)) != 0:
                            G = i * g1 + j * g2 + k * g3
                            GX = np.sum(G * dX)
                            G2 = np.sum(G**2)
                            # Add the reciprocal space contribution
                            x = 2 * np.pi / Omega * np.exp(-0.25 * G2 / nu**2) / G2
                            Eewald += x * ZiZj * np.cos(GX)
    return Eewald


def get_n_single(atoms, Y):
    '''Calculate the single-electron densities.

    Args:
        atoms: Atoms object.
        Y (array): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        array: Single-electron densities.
    '''
    Yrs = atoms.I(Y)
    return atoms.f * np.real(Yrs.conj() * Yrs)


def get_Esic(atoms, Y, n_single=None):
    '''Calculate the Perdew-Zunger self-interaction energy.

    Reference: Phys. Rev. B 23, 5048.

    Args:
        atoms: Atoms object.
        Y (array): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n_single(array): Single-electron densities.

    Returns:
        float: PZ self-interaction energy.
    '''
    # E_PZ-SIC = \sum_i Ecoul[n_i] + Exc[n_i, 0]
    if n_single is None:
        n_single = get_n_single(atoms, Y)
    Esic = 0
    for i in range(atoms.Ns):
        # Normalize single-particle densities to 1
        ni = n_single[:, i] / atoms.f[i]
        coul = get_Ecoul(atoms, ni)
        # The exchange part for a SIC correction has to be spin polarized
        xc = get_Exc(atoms, ni, spinpol=True)
        # SIC energy is scaled by the occupation number
        Esic += (coul + xc) * atoms.f[i]
    atoms.energies.Esic = Esic
    return Esic
