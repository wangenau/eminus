# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Calculate different energy contributions."""

import dataclasses

import numpy as np
from scipy.linalg import inv, norm
from scipy.special import erfc

from .dft import get_n_single, get_phi, H
from .extras import dispersion
from .gga import get_grad_field, get_tau
from .logger import log
from .tools import electronic_entropy
from .utils import handle_k
from .xc import get_exc


@dataclasses.dataclass
class Energy:
    """Energy class to save energy contributions in one place."""

    Ekin: float = 0  #: Kinetic energy.
    Ecoul: float = 0  #: Coulomb energy.
    Exc: float = 0  #: Exchange-correlation energy.
    Eloc: float = 0  #: Local energy.
    Enonloc: float = 0  #: Non-local energy.
    Eewald: float = 0  #: Ewald energy.
    Esic: float = 0  #: Self-interaction correction energy.
    Edisp: float = 0  #: Dispersion correction energy.
    Eentropy: float = 0  #: Fillings entropic energy.

    @property
    def Etot(self):
        """Total energy is the sum of all energy contributions."""
        return sum(getattr(self, ie.name) for ie in dataclasses.fields(self))

    def extrapolate(self):
        """Calculate the total energy at T=0.

        Reference: J. Phys.: Condens. Matter 1, 689.

        Returns:
            Total energy extrapolated to T=0.
        """
        return self.Etot - 0.5 * self.Eentropy

    def __repr__(self):
        """Print the energies stored in the Energy object."""
        out = ""
        for ie in dataclasses.fields(self):
            energy = getattr(self, ie.name)
            if energy != 0:
                out += f"{ie.name:<9}: {energy:+.9f} Eh\n"
        return f"{out}{'-' * 26}\nEtot     : {self.Etot:+.9f} Eh"


def get_E(scf):
    """Calculate energy contributions and update energies needed in one SCF step.

    Args:
        scf: SCF object.

    Returns:
        Total energy.
    """
    scf.energies.Ekin = get_Ekin(scf.atoms, scf.Y)
    scf.energies.Ecoul = get_Ecoul(scf.atoms, scf.n, scf.phi)
    scf.energies.Exc = get_Exc(scf, scf.n, scf.exc, Nspin=scf.atoms.occ.Nspin)
    scf.energies.Eloc = get_Eloc(scf, scf.n)
    scf.energies.Enonloc = get_Enonloc(scf, scf.Y)
    return scf.energies.Etot


@handle_k(mode="reduce")
def get_Ekin(atoms, Y, ik):
    """Calculate the kinetic energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.
        ik: k-point index.

    Returns:
        Kinetic energy in Hartree.
    """
    # Ekin = -0.5 Tr(F Wdag L(W))
    Ekin = 0
    for spin in range(atoms.occ.Nspin):
        Ekin += (
            -0.5
            * atoms.kpts.wk[ik]
            * np.trace(atoms.occ.F[ik][spin] @ Y[spin].conj().T @ atoms.L(Y[spin], ik))
        )
    return np.real(Ekin)


def get_Ecoul(atoms, n, phi=None):
    """Calculate the Coulomb energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        atoms: Atoms object.
        n: Real-space electronic density.

    Keyword Args:
        phi: Hartree field.

    Returns:
        Coulomb energy in Hartree.
    """
    if phi is None:
        phi = get_phi(atoms, n)
    # Ecoul = 0.5 (J(n))dag O(phi)
    return np.real(0.5 * n @ atoms.Jdag(atoms.O(phi)))


def get_Exc(scf, n, exc=None, n_spin=None, dn_spin=None, tau=None, Nspin=2):
    """Calculate the exchange-correlation energy.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        n: Real-space electronic density.

    Keyword Args:
        exc: Exchange-correlation energy density.
        n_spin: Real-space electronic densities per spin channel.
        dn_spin: Real-space gradient of densities per spin channel.
        tau: Real-space kinetic energy densities per spin channel.
        Nspin: Number of spin states.

    Returns:
        Exchange-correlation energy in Hartree.
    """
    atoms = scf.atoms
    if exc is None:
        exc = get_exc(scf.xc, n_spin, Nspin, dn_spin, tau, scf.xc_params)
    # Exc = (J(n))dag O(J(exc))
    return np.real(n @ atoms.Jdag(atoms.O(atoms.J(exc))))


def get_Eloc(scf, n):
    """Calculate the local energy contribution.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        n: Real-space electronic density.

    Returns:
        Local energy contribution in Hartree.
    """
    return np.real(np.vdot(scf.Vloc, n))


@handle_k(mode="reduce")
def get_Enonloc(scf, Y, ik):
    """Calculate the non-local GTH energy contribution.

    Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/calc_energies.jl

    Reference: Phys. Rev. B 54, 1703.

    Args:
        scf: SCF object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.
        ik: k-point index.

    Returns:
        Non-local GTH energy contribution in Hartree.
    """
    atoms = scf.atoms

    Enonloc = 0
    if scf.pot != "gth" or scf.gth.NbetaNL == 0:  # Only calculate the non-local part if necessary
        return Enonloc

    for spin in range(atoms.occ.Nspin):
        betaNL_psi = (Y[spin].conj().T @ scf.gth.betaNL[ik]).conj()

        enl = np.zeros(Y.shape[-1], dtype=complex)
        for ia in range(atoms.Natoms):
            psp = scf.gth[atoms.atom[ia]]
            for l in range(psp["lmax"]):
                for m in range(-l, l + 1):
                    for iprj in range(psp["Nproj_l"][l]):
                        ibeta = scf.gth.prj2beta[iprj, ia, l, m + psp["lmax"] - 1] - 1
                        for jprj in range(psp["Nproj_l"][l]):
                            jbeta = scf.gth.prj2beta[jprj, ia, l, m + psp["lmax"] - 1] - 1
                            hij = psp["h"][l, iprj, jprj]
                            enl += hij * betaNL_psi[:, ibeta].conj() * betaNL_psi[:, jbeta]
        Enonloc += np.sum(atoms.occ.f[ik, spin] * atoms.kpts.wk[ik] * enl)
    # We have to multiply with the cell volume, because of different orthogonalization methods
    return np.real(Enonloc * atoms.Omega)


def get_Eewald(atoms, gcut=2, gamma=1e-8):
    """Calculate the Ewald energy.

    Reference: J. Chem. Theory Comput. 10, 381.

    Args:
        atoms: Atoms object.

    Keyword Args:
        gcut: G-vector cut-off.
        gamma: Error tolerance.

    Returns:
        Ewald energy in Hartree.
    """

    # For a plane wave code we have multiple contributions for the Ewald energy
    # Namely, a sum from contributions from real-space, reciprocal space,
    # the self energy, (the dipole term [neglected]), and an additional electroneutrality term
    def get_index_vectors(s):
        """Create index vectors of s periodic images."""
        m1 = np.arange(-s[0], s[0] + 1)
        m2 = np.arange(-s[1], s[1] + 1)
        m3 = np.arange(-s[2], s[2] + 1)
        M = np.transpose(np.meshgrid(m1, m2, m3)).reshape(-1, 3)
        # Delete the [0, 0, 0] element, to prevent checking for it in every loop
        return M[~(M == 0).all(axis=1)]

    # Calculate the Ewald splitting parameter nu
    gexp = -np.log(gamma)
    nu = 0.5 * np.sqrt(gcut**2 / gexp)

    # Start by calculating the self-energy
    Eewald = -nu / np.sqrt(np.pi) * np.sum(atoms.Z**2)
    # Add the electroneutrality term
    Eewald += -np.pi * np.sum(atoms.Z) ** 2 / (2 * nu**2 * atoms.Omega)

    # Calculate the real-space contribution
    # Calculate the amount of images that have to be considered per axis
    Rm = norm(atoms.a, axis=1)
    tmax = np.sqrt(0.5 * gexp) / nu
    s = np.rint(tmax / Rm + 1.5)
    # Collect all box index vectors in a matrix
    M = get_index_vectors(s)
    # Calculate the translation vectors
    T = M @ atoms.a

    # Calculate the reciprocal space contribution
    # Calculate the amount of reciprocal images that have to be considered per axis
    g = 2 * np.pi * inv(atoms.a.T)
    gm = norm(g, axis=1)
    s = np.rint(gcut / gm + 1.5)
    # Collect all box index vectors in a matrix
    M = get_index_vectors(s)
    # Calculate the reciprocal translation vectors and precompute the prefactor
    G = M @ g
    G2 = norm(G, axis=1) ** 2
    prefactor = 2 * np.pi / atoms.Omega * np.exp(-0.25 * G2 / nu**2) / G2

    for ia in range(atoms.Natoms):
        for ja in range(atoms.Natoms):
            dpos = atoms.pos[ia] - atoms.pos[ja]
            ZiZj = atoms.Z[ia] * atoms.Z[ja]

            # Add the real-space contribution
            rmag = norm(dpos - T, axis=1)
            Eewald += 0.5 * ZiZj * np.sum(erfc(rmag * nu) / rmag)
            # The T=[0, 0, 0] element is omitted in M but needed if ia!=ja, so add it manually
            if ia != ja:
                rmag = norm(dpos)
                Eewald += 0.5 * ZiZj * erfc(rmag * nu) / rmag

            # Add the reciprocal space contribution
            Gpos = np.sum(G * dpos, axis=1)
            Eewald += ZiZj * np.sum(prefactor * np.cos(Gpos))
    return Eewald


def get_Esic(scf, Y, n_single=None):
    """Calculate the Perdew-Zunger self-interaction energy.

    Reference: Phys. Rev. B 23, 5048.

    Args:
        scf: SCF object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        n_single: Single-electron densities.

    Returns:
        PZ self-interaction energy.
    """
    if Y is None:
        log.warning('The provided wave function is "None".')
        return None

    atoms = scf.atoms
    # E_PZ-SIC = \sum_i Ecoul[n_i] + Exc[n_i, 0]
    if n_single is None:
        n_single = get_n_single(atoms, Y)

    Esic = 0
    for i in range(atoms.occ.Nstate):
        for spin in range(atoms.occ.Nspin):
            if np.sum(atoms.occ.f[:, spin, i] * atoms.kpts.wk) > 0:
                # Create normalized single-particle densities in the form of electronic densities
                # per spin channel, since spin-polarized functionals expect this form
                ni = np.zeros((2, atoms.Ns))
                # Normalize single-particle densities to 1
                ni[0] = n_single[spin, :, i] / np.sum(atoms.occ.f[:, spin, i] * atoms.kpts.wk)

                # Get the gradient of the single-particle density
                if "gga" in scf.xc_type:
                    dni = np.zeros((2, atoms.Ns, 3))
                    dni[0] = get_grad_field(atoms, ni)[0]
                else:
                    dni = None

                # Get the kinetic energy density of the corresponding orbital
                if scf.xc_type == "meta-gga":
                    # Use only one orbital for the calculation
                    Ytmp = []
                    for ik in range(atoms.kpts.Nk):
                        Ytmp.append(np.zeros_like(Y[ik]))
                        Ytmp[ik][0, :, 0] = Y[ik][spin, :, i]
                    taui = np.zeros_like(ni)
                    # We also have to normalize to one again
                    taui[0] = get_tau(atoms, Ytmp)[0] / np.sum(
                        atoms.occ.f[:, spin, i] * atoms.kpts.wk
                    )
                else:
                    taui = None

                coul = get_Ecoul(atoms, ni[0])
                # The exchange part for a SIC correction has to be spin-polarized
                xc = get_Exc(scf, ni[0], n_spin=ni, dn_spin=dni, tau=taui, Nspin=2)
                # SIC energy is scaled by the occupation number
                Esic += (coul + xc) * np.sum(atoms.occ.f[:, spin, i] * atoms.kpts.wk)
    scf.energies.Esic = Esic
    return Esic


def get_Edisp(scf, version="d3bj", atm=True, xc=None):  # noqa: D103
    try:
        return dispersion.get_Edisp(scf, version, atm, xc)
    except ImportError:
        log.warning("You have to install the dispersion extra to use this function.")
        return 0


get_Edisp.__doc__ = dispersion.get_Edisp.__doc__


def get_Eband(scf, Y, **kwargs):
    """Calculate the band energy for occupied or unoccupied bands.

    Reference: Comput. Phys. Commun. 128, 1.

    Args:
        scf: SCF object.
        Y: Expansion coefficients of orthogonal wave functions in reciprocal space.

    Keyword Args:
        **kwargs: See :func:`eminus.dft.H`.

    Returns:
        Band energy in Hartree.
    """
    atoms = scf.atoms
    # Eband = Tr(Wdag H(W))
    Eband = 0
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            Eband += atoms.kpts.wk[ik] * np.trace(
                Y[ik][spin].conj().T @ H(scf, ik, spin, Y, **kwargs)
            )
    return np.real(Eband)


def get_Eentropy(scf, epsilon, Efermi):
    """Calculate the fillings entropic energy.

    Reference: J. Phys. Condens. Matter 1, 689.

    Args:
        scf: SCF object.
        epsilon: Eigenenergies.
        Efermi: Fermi energy.

    Returns:
        Entropic energy in Hartree.
    """
    occ = scf.atoms.occ

    Eentropy = 0
    for ik in range(scf.kpts.Nk):
        for spin in range(occ.Nspin):
            for i in range(occ.Nstate):
                # Beware the sign change, it is handled in the electronic_entropy function
                Eentropy += (
                    occ.wk[ik]
                    * occ.smearing
                    * electronic_entropy(epsilon[ik, spin, i], Efermi, occ.smearing)
                )

    Eentropy *= 2 / occ.Nspin
    scf.energies.Eentropy = Eentropy
    return Eentropy
