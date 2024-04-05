#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Various tools to check physical properties."""

import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize_scalar, root_scalar

from .dft import get_epsilon, get_epsilon_unocc
from .gga import get_grad_field, get_tau
from .logger import log
from .utils import handle_k_gracefully, skip_k


def cutoff2gridspacing(E):
    """Convert plane wave energy cut-off to a real-space grid spacing.

    Reference: Phys. Rev. B 54, 14362.

    Args:
        E (float): Energy in Hartree.

    Returns:
        float: Grid spacing in Bohr.
    """
    return np.pi / np.sqrt(2 * E)


def gridspacing2cutoff(h):
    """Convert real-space grid spacing to plane wave energy cut-off.

    Reference: Phys. Rev. B 54, 14362.

    Args:
        h (float): Grid spacing in Bohr.

    Returns:
        float: Cut-off in Hartree.
    """
    return 0.5 * (np.pi / h) ** 2


def center_of_mass(coords, masses=None):
    """Calculate the center of mass for a set of coordinates and masses.

    Args:
        coords (ndarray): Array of real-space coordinates.

    Keyword Args:
        masses (ndarray | None): Mass or weight for each coordinate.

    Returns:
        ndarray: Center of mass.
    """
    if masses is None:
        masses = np.ones(len(coords))
    return np.sum(masses * coords.T, axis=1) / np.sum(masses)


@handle_k_gracefully
def orbital_center(obj, psirs):
    """Calculate the orbital center of masses, e.g., from localized orbitals.

    Args:
        obj: Atoms or SCF object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        bool: Center of masses.
    """
    atoms = obj._atoms

    coms = [np.array([])] * 2
    Ncom = psirs.shape[2]
    for spin in range(atoms.occ.Nspin):
        coms_spin = np.empty((Ncom, 3))

        # Squared orbitals
        psi2 = np.real(psirs[spin].conj() * psirs[spin])
        for i in range(Ncom):
            coms_spin[i] = center_of_mass(atoms.r, psi2[:, i])
        coms[spin] = coms_spin
    return coms


def inertia_tensor(coords, masses=None):
    """Calculate the inertia tensor for a set of coordinates and masses.

    Reference: https://en.wikipedia.org/wiki/Moment_of_inertia

    Args:
        coords (ndarray): Array of real-space coordinates.

    Keyword Args:
        masses (ndarray | None): Mass or weight for each coordinate.

    Returns:
        ndarray: Inertia tensor.
    """
    if masses is None:
        masses = np.ones(len(coords))

    # The inertia tensor for a set of point masses can be calculated with a simple summation
    # https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    I = np.empty((3, 3))
    I[0, 0] = np.sum(masses * (coords[:, 1] ** 2 + coords[:, 2] ** 2))
    I[1, 1] = np.sum(masses * (coords[:, 0] ** 2 + coords[:, 2] ** 2))
    I[2, 2] = np.sum(masses * (coords[:, 0] ** 2 + coords[:, 1] ** 2))
    I[0, 1] = I[1, 0] = -np.sum(masses * (coords[:, 0] * coords[:, 1]))
    I[0, 2] = I[2, 0] = -np.sum(masses * (coords[:, 0] * coords[:, 2]))
    I[1, 2] = I[2, 1] = -np.sum(masses * (coords[:, 1] * coords[:, 2]))
    return I


def get_dipole(scf, n=None):
    """Calculate the electric dipole moment.

    This function does not account for periodicity, it may be a good idea to center the system.

    Reference: J. Chem. Phys. 155, 224109.

    Args:
        scf: SCF object.

    Keyword Args:
        n (float | None): Real-space electronic density.

    Returns:
        ndarray: Electric dipole moment in e * Bohr.
    """
    atoms = scf.atoms
    if n is None:
        if not hasattr(scf, 'n') or scf.n is None:
            log.error('There is no density to calculate a dipole moment.')
            return 0
        n = scf.n

    # Diple moment: mu = \sum Z pos - \int n(r) r dr
    mu = np.zeros(3)
    for i in range(atoms.Natoms):
        mu += atoms.Z[i] * atoms.pos[i]

    for dim in range(3):
        mu[dim] -= atoms.dV * np.sum(n * atoms.r[:, dim])
    return mu


def get_ip(scf):
    """Calculate the ionization potential by calculating the negative HOMO energy.

    Reference: Physica 1, 104.

    Args:
        scf: SCF object.

    Returns:
        float: Ionization potential in Hartree.
    """
    scf.atoms.kpts._assert_gamma_only()
    epsilon = get_epsilon(scf, scf.W)[0]
    # Account for spin-polarized calculations
    epsilon = np.sort(np.ravel(epsilon))
    return -epsilon[-1]


@skip_k
def check_ortho(obj, func, eps=1e-9):
    """Check the orthogonality condition for a set of functions.

    Args:
        obj: Atoms or SCF object.
        func (ndarray): A discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Orthogonality status for the set of functions.
    """
    atoms = obj._atoms
    func = np.atleast_3d(func)

    # It makes no sense to calculate anything for only one function
    if atoms.occ.Nstate == 1:
        log.warning('Need at least two functions to check their orthogonality.')
        return True

    ortho_bool = True
    # Check the condition for every combination
    # Orthogonality condition: \int func1^* func2 dr = 0
    for spin in range(atoms.occ.Nspin):
        for i in range(atoms.occ.Nstate):
            for j in range(i + 1, atoms.occ.Nstate):
                res = atoms.dV * np.sum(func[spin, :, i].conj() * func[spin, :, j])
                tmp_bool = abs(res) < eps
                ortho_bool *= tmp_bool
                log.debug(f'Function {i} and {j}:\nValue: {res:.7f}\nOrthogonal: {tmp_bool}')
    log.info(f'Orthogonal: {ortho_bool}')
    return ortho_bool


@skip_k
def check_norm(obj, func, eps=1e-9):
    """Check the normalization condition for a set of functions.

    Args:
        obj: Atoms or SCF object.
        func (ndarray): A discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Normalization status for the set of functions.
    """
    atoms = obj._atoms
    func = np.atleast_3d(func)

    norm_bool = True
    # Check the condition for every function
    # Normality condition: \int func^* func dr = 1
    for spin in range(atoms.occ.Nspin):
        for i in range(atoms.occ.Nstate):
            res = atoms.dV * np.sum(func[spin, :, i].conj() * func[spin, :, i])
            tmp_bool = abs(1 - res) < eps
            norm_bool *= tmp_bool
            log.debug(f'Function {i}:\nValue: {res:.7f}\nNormalized: {tmp_bool}')
    log.info(f'Normalized: {norm_bool}')
    return norm_bool


@skip_k
def check_orthonorm(obj, func, eps=1e-9):
    """Check the orthonormality conditions for a set of functions.

    Args:
        obj: Atoms or SCF object.
        func (ndarray): A discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Orthonormality status for the set of functions.
    """
    atoms = obj._atoms
    ortho_bool = check_ortho(atoms, func, eps)
    norm_bool = check_norm(atoms, func, eps)
    log.info(f'Orthonormal: {ortho_bool * norm_bool}')
    return ortho_bool * norm_bool


def get_isovalue(n, percent=85):
    """Find an isovalue that contains a percentage of the electronic density.

    Reference: J. Chem. Phys. 158, 164102.

    Args:
        n (float): Real-space electronic density.

    Keyword Args:
        percent (float): Amount of density that should be contained.

    Returns:
        float: Isovalue that contains the specified percentage of the density.
    """

    def deviation(isovalue):
        """Wrapper function for finding the isovalue by minimization."""
        n_mask = np.sum(n[n > isovalue])
        return abs(percent - (n_mask / n_ref) * 100)

    # Integrated density
    n_ref = np.sum(n)
    # Finding the isovalue is an optimization problem, minimizing the deviation above
    # The problem is bound by zero (no density) and the maximum value in n
    res = minimize_scalar(deviation, bounds=(0, np.max(n)), method='bounded')
    return res.x


def get_tautf(scf):
    """Calculate the Thomas-Fermi kinetic energy densities per spin.

    Reference: Phys. Lett. B 63, 395.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Real-space Thomas-Fermi kinetic energy density.
    """
    atoms = scf.atoms
    # Use the definition with a division by two
    tautf = 3 / 10 * (atoms.occ.Nspin * 3 * np.pi**2) ** (2 / 3) * scf.n_spin ** (5 / 3)

    log.debug(f'Calculated Ekin:  {scf.energies.Ekin:.6f} Eh')
    log.debug(f'Integrated tautf: {np.sum(tautf) * atoms.dV:.6f} Eh')
    return tautf


def get_tauw(scf):
    """Calculate the von Weizsaecker kinetic energy densities per spin.

    Reference: Z. Phys. 96, 431.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Real-space von Weizsaecker kinetic energy density.
    """
    atoms = scf.atoms
    if scf.dn_spin is None:
        dn_spin = get_grad_field(atoms, scf.n_spin)
    else:
        dn_spin = scf.dn_spin
    dn2 = norm(dn_spin, axis=2) ** 2
    # Use the definition with a division by two
    tauw = dn2 / (8 * scf.n_spin)

    # For one- and two-electron systems the integrated KED has to be the same as the calculated KE
    log.debug(f'Calculated Ekin: {scf.energies.Ekin:.6f} Eh')
    log.debug(f'Integrated tauw: {np.sum(tauw) * atoms.dV:.6f} Eh')
    return tauw


def get_elf(scf):
    """Calculate the electron localization function.

    Reference: J. Chem. Phys. 92, 5397.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Real-space electron localization function.
    """
    D = get_tau(scf.atoms, scf.Y) - get_tauw(scf)
    D0 = get_tautf(scf)
    X = D / D0
    return 1 / (1 + X**2)


def get_reduced_gradient(scf, eps=0):
    """Calculate the reduced density gradient s.

    Reference: Phys. Rev. Lett. 78, 1396.

    Args:
        scf: SCF object.

    Kwargs:
        eps (float): Threshold of the density where s should be truncated.

    Returns:
        ndarray: Real-space reduced density gradient.
    """
    atoms = scf.atoms
    if scf.dn_spin is None:
        dn_spin = get_grad_field(atoms, scf.n_spin)
    else:
        dn_spin = scf.dn_spin
    norm_dn = norm(np.sum(dn_spin, axis=0), axis=1)

    kf = (3 * np.pi**2 * scf.n) ** (1 / 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = norm_dn / (2 * kf * scf.n)
    s[scf.n < eps] = 0
    return s


def get_spin_squared(scf):
    """Calculate the expectation value of the squared spin operator <S^2>.

    Reference: Appl. Phys. Express 12, 115506.

    Args:
        scf: SCF object.

    Returns:
        float: The DFT value for <S^2>.
    """
    atoms = scf.atoms
    # <S^2> for a restricted calculation is always zero
    if not atoms.unrestricted:
        return 0

    rhoXr = scf.n_spin[0] - scf.n_spin[1]
    rhoXr[rhoXr < 0] = 0
    rhoX = np.sum(rhoXr) * atoms.dV
    SX = 0.5 * (np.sum(scf.n_spin[0]) - np.sum(scf.n_spin[1])) * atoms.dV
    return SX * (SX + 1) + rhoX


def get_multiplicity(scf):
    """Calculate the multiplicity from <S^2>.

    Args:
        scf: SCF object.

    Returns:
        float: Multiplicity 2S+1.
    """
    S2 = get_spin_squared(scf)
    # <S^2> = S(S+1) = S^2+S+0.25-0.25 = (S+0.5)^2-0.25 => S = sqrt(<S^2>+0.25)-0.5
    S = np.sqrt(S2 + 0.25) - 0.5
    return 2 * S + 1


def get_bandgap(scf):
    """Calculate the band gap.

    Args:
        scf: SCF object.

    Returns:
        float: Band gap energy.
    """
    e_occ = get_epsilon(scf, scf.W, **scf._precomputed)

    if not hasattr(scf, 'Z') or scf.Z is None:
        log.warning("The SCF object has no unoccupied energies, can't calculate band gap.")
        return 0

    e_unocc = get_epsilon_unocc(scf, scf.W, scf.Z, **scf._precomputed)
    return np.min(e_unocc) - np.max(e_occ)


def get_Efermi(obj, epsilon=None):
    """Calculate the Fermi energy.

    Reference: Phys. Rev. B 107, 195122.

    Args:
        obj: SCF or Occupations object.

    Keyword Args:
        epsilon (ndarray): Eigenenergies.

    Returns:
        float: Fermi energy.
    """
    # Handle the obj argument
    if hasattr(obj, 'smearing'):
        if epsilon is None:
            log.error('When passing an Occupations object the eigenenergies have to be given.')
        occ = obj
    else:
        occ = obj.atoms.occ

    # Calculate the eigenenergies if necessary
    if epsilon is None:
        e_occ = get_epsilon(obj, obj.W, **obj._precomputed)
    else:
        e_occ = epsilon

    def electron_root(Efermi):
        """Number of electrons by Fermi distribution minus the actual number of electrons."""
        occ_sum = 0
        for ik in range(occ.Nk):
            occ_sum += occ.wk[ik] * np.sum(fermi_distribution(e_occ[ik], Efermi, occ.smearing))
        return occ_sum * 2 / occ.Nspin - occ.Nelec

    # For smeared systems we have to find the root of an objective function
    if occ.smearing > 0:
        return root_scalar(electron_root, bracket=(np.min(e_occ), np.max(e_occ))).root

    if not hasattr(obj, 'Z') or obj.Z is None:
        log.warning('The SCF object has no unoccupied energies, return the maximum energy instead.')
        return np.max(e_occ)

    e_unocc = get_epsilon_unocc(obj, obj.W, obj.Z, **obj._precomputed)
    return np.max(e_occ) + (np.min(e_unocc) - np.max(e_occ)) / 2


def fermi_distribution(E, mu, kbT):
    """Calculate the Fermi distribution.

    Reference: https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics

    Args:
        E (float): State energy.
        mu (float): Chemical energy or Fermi energy.
        kbT (float): Thermic energy or smearing width.

    Returns:
        float: Fermi distribution.
    """
    x = (E - mu) / kbT
    with np.errstate(over='ignore'):
        return 1 / (np.exp(x) + 1)


def electronic_entropy(E, mu, kbT):
    """Calculate the electronic entropic energy.

    Reference: J. Phys. Condens. Matter 1, 689.

    Args:
        E (float): State energy.
        mu (float): Chemical energy or Fermi energy.
        kbT (float): Thermic energy or smearing width.

    Returns:
        float: Electronic entropic energy.
    """
    # Condition taken from: https://gitlab.com/QEF/q-e/-/blob/master/Modules/w1gauss.f90
    if abs((E - mu) / kbT) > 36:
        return 0
    f = fermi_distribution(E, mu, kbT)
    return f * np.log(f) + (1 - f) * np.log(1 - f)
