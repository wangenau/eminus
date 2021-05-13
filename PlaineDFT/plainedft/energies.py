#!/usr/bin/env python3
'''
Calculate energies for a basis set or one-particle densities.
'''
import numpy as np
from numpy.linalg import det, norm, inv
from scipy.special import erfc
from .lda_VWN import excVWN, xc_vwn
from .tools import ry2ha


def get_Ekin(atoms, Y):
    '''Calculate the kinetic energy.'''
    # Arias: Ekin = -0.5 Tr(F Cdag L(C))
    return np.real(-0.5 * np.trace(np.diag(atoms.f) @ (Y.conj().T @ atoms.L(Y))))


def get_Ecoul(atoms, n):
    '''Calculate the coulomb energy.'''
    # Arias: Ecoul = -(Jn)dag O(phi)
    phi = -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
    return np.real(0.5 * n.conj().T @ atoms.Jdag(atoms.O(phi)))


def get_Exc(atoms, n):
    '''Calculate the exchange-correlation energy.'''
    # Arias: Exc = (Jn)dag O(J(exc))
    exc = excVWN(n)
    return np.real(n.conj().T @ atoms.Jdag(atoms.O(atoms.J(exc))))
    #return np.dot(xc_vwn(n)[0], n) * atoms.CellVol / np.prod(atoms.S)


def get_Eloc(atoms, n):
    '''Calculate the local energy.'''
    return np.real(atoms.Vloc.conj().T @ n)# * atoms.CellVol / np.prod(atoms.S)


def get_Esic(atoms, n):
    '''Calculate the PErdew-Zunger self-interaction energy.'''
    # E_PZ-SIC = \sum_i Ecoul[n_i] + Exc[n_i, 0]
    Esic = 0
    for i in range(len(n)):
        coul = get_Ecoul(atoms, n[i])
        xc = get_Exc(atoms, n[i])
        Esic += coul + xc
    return Esic


# Adopted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_energies.jl
def get_Enonloc(atoms, W):
    '''Calculate the non-local energy.'''
    Enonloc = 0
    if atoms.NbetaNL > 0:
        # Parameters of the non-local pseudopotential
        prj2beta = atoms.prj2beta
        betaNL = atoms.betaNL

        Natoms = len(atoms.X)
        Nstates = atoms.Ns

        betaNL_psi = np.dot(W.T.conj(), betaNL).conj()

        Enonloc = 0
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
    return Enonloc


# Adopted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_E_NN.jl
def get_Eewald(atoms):
    '''Calculate the Ewald energy.'''
    # For a plane wave code we have multiple contributions for the Ewald energy
    # Namely, a sum from contributions from real space, reciprocal space,
    # the self energy, (the dipole term [neglected]), and an additional electroneutrality-term
    # See Eq. (4) https://juser.fz-juelich.de/record/16155/files/IAS_Series_06.pdf
    # Note: This code calculates the energy in Rydberg, so the equations are off
    # ba a factor 2
    Natoms = len(atoms.X)
    tau = np.asarray(atoms.X)
    Zvals = np.asarray(atoms.Z)
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

    gcut = 2
    ebsl = 1e-8
    gexp = -np.log(ebsl)
    nu = 0.5 * np.sqrt(gcut**2 / gexp)

    tmax = np.sqrt(0.5 * gexp) / nu
    mmm1 = int(np.rint(tmax / t1m + 1.5))
    mmm2 = int(np.rint(tmax / t2m + 1.5))
    mmm3 = int(np.rint(tmax / t3m + 1.5))

    # Start by calculaton the self-energy
    ewald = -2 * nu * x / np.sqrt(np.pi)
    # Add the electroneutrality-term (Eq. 11)
    ewald += -np.pi * totalcharge**2 / (omega * nu**2)

    dtau = np.zeros(3)
    G = np.zeros(3)
    T = np.zeros(3)
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
                            ewald += ZiZj * erfc(rmag * nu) / rmag

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
                            x = 4 * np.pi / omega * np.exp(-0.25 * G2 / nu**2) / G2
                            ewald += x * ZiZj * np.cos(Gtau)

    return ry2ha(ewald)  # Convert to Hartree
