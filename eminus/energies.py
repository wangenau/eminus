#!/usr/bin/env python3
'''Calculate different energy contributions.'''
import dataclasses

import numpy as np
from scipy.linalg import inv, norm
from scipy.special import erfc

from .dft import get_grad_field, get_n_single, solve_poisson
from .xc import get_exc


@dataclasses.dataclass
class Energy:
    '''Energy class to save energy contributions in one place.'''
    Ekin: float = 0     #: Kinetic energy.
    Ecoul: float = 0    #: Coulomb energy.
    Exc: float = 0      #: Exchange-correlation energy.
    Eloc: float = 0     #: Local energy.
    Enonloc: float = 0  #: Non-local energy.
    Eewald: float = 0   #: Ewald energy.
    Esic: float = 0     #: Self-interaction correction energy.

    @property
    def Etot(self):
        '''Total energy is the sum of all energy contributions.'''
        return sum(getattr(self, ie.name) for ie in dataclasses.fields(self))

    def __repr__(self):
        '''Print the energies stored in the Energy object.'''
        out = ''
        for ie in dataclasses.fields(self):
            energy = getattr(self, ie.name)
            if energy != 0:
                out += f'{ie.name.ljust(8)}: {energy:+.9f} Eh\n'
        return f'{out}{"-" * 25}\nEtot    : {self.Etot:+.9f} Eh'


def get_E(scf):
    '''Calculate energy contributions and update energies needed in one SCF step.

    Args:
        scf: SCF object.

    Returns:
        float: Total energy.
    '''
    scf.energies.Ekin = get_Ekin(scf.atoms, scf.Y)
    scf.energies.Ecoul = get_Ecoul(scf.atoms, scf.n, scf.phi)
    scf.energies.Exc = get_Exc(scf, scf.n, scf.exc, scf.atoms.Nspin)
    scf.energies.Eloc = get_Eloc(scf, scf.n)
    scf.energies.Enonloc = get_Enonloc(scf, scf.Y)
    return scf.energies.Etot


def get_Ekin(atoms, Y):
    '''Calculate the kinetic energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        float: Kinetic energy in Hartree.
    '''
    # Ekin = -0.5 Tr(F Wdag L(W))
    Ekin = 0
    for spin in range(atoms.Nspin):
        F = np.diag(atoms.f[spin])
        Ekin += -0.5 * np.trace(F @ Y[spin].conj().T @ atoms.L(Y[spin]))
    return np.real(Ekin)


def get_Ecoul(atoms, n, phi=None):
    '''Calculate the Coulomb energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n (ndarray): Real-space electronic density.

    Keyword Args:
        phi (ndarray): Hartree ﬁeld.

    Returns:
        float: Coulomb energy in Hartree.
    '''
    if phi is None:
        phi = solve_poisson(atoms, n)
    # Ecoul = 0.5 (J(n))dag O(phi)
    return np.real(0.5 * n @ atoms.Jdag(atoms.O(phi)))


def get_Exc(scf, n, exc=None, n_spin=None, dn_spin=None, Nspin=2):
    '''Calculate the exchange-correlation energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        n (ndarray): Real-space electronic density.

    Keyword Args:
        exc (ndarray): Exchange-correlation energy density.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        dn_spin (ndarray): Real-space gradient of densities per spin channel.
        Nspin (int): Number of spin states.

    Returns:
        float: Exchange-correlation energy in Hartree.
    '''
    atoms = scf.atoms
    if exc is None:
        if dn_spin is None and 'gga' in scf.xc_type:
            dn_spin = get_grad_field(atoms, n_spin)
        exc = get_exc(scf.xc, n_spin, Nspin, dn_spin)
    # Exc = (J(n))dag O(J(exc))
    return np.real(n @ atoms.Jdag(atoms.O(atoms.J(exc))))


def get_Eloc(scf, n):
    '''Calculate the local energy contribution.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        n (ndarray): Real-space electronic density.

    Returns:
        float: Local energy contribution in Hartree.
    '''
    return np.real(scf.Vloc.conj().T @ n)


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_energies.jl
def get_Enonloc(scf, Y):
    '''Calculate the non-local GTH energy contribution.

    Reference: Phys. Rev. B 54, 1703.

    Args:
        scf: SCF object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Returns:
        float: Non-local GTH energy contribution in Hartree.
    '''
    atoms = scf.atoms

    Enonloc = 0
    if scf.NbetaNL > 0:  # Only calculate non-local potential if necessary
        for spin in range(atoms.Nspin):
            betaNL_psi = (Y[spin].conj().T @ scf.betaNL).conj()

            enl = np.zeros(atoms.Nstate, dtype=complex)
            for ia in range(atoms.Natoms):
                psp = scf.GTH[atoms.atom[ia]]
                for l in range(psp['lmax']):
                    for m in range(-l, l + 1):
                        for iprj in range(psp['Nproj_l'][l]):
                            ibeta = scf.prj2beta[iprj, ia, l, m + psp['lmax'] - 1] - 1
                            for jprj in range(psp['Nproj_l'][l]):
                                jbeta = scf.prj2beta[jprj, ia, l, m + psp['lmax'] - 1] - 1
                                hij = psp['h'][l, iprj, jprj]
                                enl += hij * betaNL_psi[:, ibeta].conj() * betaNL_psi[:, jbeta]
            Enonloc += np.sum(atoms.f[spin] * enl)
    # We have to multiply with the cell volume, because of different orthogonalization methods
    return np.real(Enonloc * atoms.Omega)


def get_Eewald(atoms, gcut=2, gamma=1e-8):
    '''Calculate the Ewald energy.

    Reference: J. Chem. Theory Comput. 10, 381.

    Args:
        atoms: Atoms object.

    Keyword Args:
        gcut (float): G-vector cut-off.
        gamma (float): Error tolerance.

    Returns:
        float: Ewald energy in Hartree.
    '''
    # For a plane wave code we have multiple contributions for the Ewald energy
    # Namely, a sum from contributions from real-space, reciprocal space,
    # the self energy, (the dipole term [neglected]), and an additional electroneutrality term
    def get_index_vectors(s):
        '''Create all index vectors of periodic images.

        Args:
            s (ndarray): Number of images per lattice vector.

        Returns:
            ndarray: Index matrix.
        '''
        m1 = np.arange(-s[0], s[0] + 1)
        m2 = np.arange(-s[1], s[1] + 1)
        m3 = np.arange(-s[2], s[2] + 1)
        M = np.transpose(np.meshgrid(m1, m2, m3)).reshape(-1, 3)
        # Delete the [0, 0, 0] element, to prevent checking for it in every loop
        return M[~np.all(M == 0, axis=1)]

    # Calculate the Ewald splitting parameter nu
    gexp = -np.log(gamma)
    nu = 0.5 * np.sqrt(gcut**2 / gexp)

    # Start by calculating the self-energy
    Eewald = -nu / np.sqrt(np.pi) * np.sum(atoms.Z**2)
    # Add the electroneutrality term
    Eewald += -np.pi * np.sum(atoms.Z)**2 / (2 * nu**2 * atoms.Omega)

    # Calculate the real-space contribution
    # Calculate the amount of images that have to be considered per axis
    Rm = norm(atoms.R, axis=1)
    tmax = np.sqrt(0.5 * gexp) / nu
    s = np.rint(tmax / Rm + 1.5)
    # Collect all box index vector in a matrix
    M = get_index_vectors(s)
    # Calculate the translation vectors
    T = M @ atoms.R

    for ia in range(atoms.Natoms):
        for ja in range(atoms.Natoms):
            dX = atoms.X[ia] - atoms.X[ja]
            ZiZj = atoms.Z[ia] * atoms.Z[ja]
            rmag = np.sqrt(norm(dX - T, axis=1)**2)
            # Add the real-space contribution
            Eewald += 0.5 * ZiZj * np.sum(erfc(rmag * nu) / rmag)
            # The T=[0, 0, 0] element is ommited in M but needed if ia!=ja, so add it manually
            if ia != ja:
                rmag = np.sqrt(norm(dX)**2)
                Eewald += 0.5 * ZiZj * erfc(rmag * nu) / rmag

    # Calculate the reciprocal space contribution
    # Calculate the amount of reciprocal images that have to be considered per axis
    g = 2 * np.pi * inv(atoms.R.T)
    gm = norm(g, axis=1)
    s = np.rint(gcut / gm + 1.5)
    # Collect all box index vector in a matrix
    M = get_index_vectors(s)
    # Calculate the reciprocal translation vectors and precompute the prefactor
    G = M @ g
    G2 = norm(G, axis=1)**2
    prefactor = 2 * np.pi / atoms.Omega * np.exp(-0.25 * G2 / nu**2) / G2

    for ia in range(atoms.Natoms):
        for ja in range(atoms.Natoms):
            dX = atoms.X[ia] - atoms.X[ja]
            ZiZj = atoms.Z[ia] * atoms.Z[ja]
            GX = np.sum(G * dX, axis=1)
            # Add the reciprocal space contribution
            Eewald += ZiZj * np.sum(prefactor * np.cos(GX))
    return Eewald


def get_Esic(scf, Y, n_single=None):
    '''Calculate the Perdew-Zunger self-interaction energy.

    Reference: Phys. Rev. B 23, 5048.

    Args:
        scf: SCF object.
        Y (ndarray): Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n_single (ndarray): Single-electron densities.

    Returns:
        float: PZ self-interaction energy.
    '''
    atoms = scf.atoms
    # E_PZ-SIC = \sum_i Ecoul[n_i] + Exc[n_i, 0]
    if n_single is None:
        n_single = get_n_single(atoms, Y)

    Esic = 0
    for i in range(atoms.Nstate):
        for spin in range(atoms.Nspin):
            if atoms.f[spin, i] > 0:
                # Create normalized single-particle densities in the form of electronic densities
                # per spin channel, since spin-polarized functionals expect this form
                ni = np.zeros((2, len(n_single[0])))
                # Normalize single-particle densities to 1
                ni[0] = n_single[spin, :, i] / atoms.f[spin, i]

                coul = get_Ecoul(atoms, ni[0])
                # The exchange part for a SIC correction has to be spin-polarized
                xc = get_Exc(scf, ni[0], n_spin=ni, Nspin=2)
                # SIC energy is scaled by the occupation number
                Esic += (coul + xc) * atoms.f[spin, i]
    scf.energies.Esic = Esic
    return Esic
