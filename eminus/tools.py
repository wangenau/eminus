#!/usr/bin/env python3
"""Various tools to check physical properties."""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize_scalar

from .dft import get_epsilon
from .gga import get_grad_field, get_tau
from .logger import log


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
    return 0.5 * (np.pi / h)**2


def center_of_mass(coords, masses=None):
    """Calculate the center of mass for a set of coordinates and masses.

    Args:
        coords (ndarray): Array of real-space coordinates.

    Keyword Args:
        masses (ndarray): Mass or weight for each coordinate.

    Returns:
        ndarray: Center of mass.
    """
    if masses is None:
        masses = np.ones(len(coords))

    return np.sum(masses * coords.T, axis=1) / np.sum(masses)


def orbital_center(object, psirs):
    """Calculate the orbital center of masses, e.g., from localized orbitals.

    Args:
        object: Atoms or SCF object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        bool: Center of masses.
    """
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object

    coms = [np.array([])] * 2
    Ncom = psirs.shape[2]
    for spin in range(atoms.Nspin):
        coms_spin = np.empty((Ncom, 3))

        # Square orbitals
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
        masses (ndarray): Mass or weight for each coordinate.

    Returns:
        ndarray: Inertia tensor.
    """
    if masses is None:
        masses = np.ones(len(coords))

    # The inertia tensor for a set of point masses can be calculated with simple summation
    # https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    I = np.empty((3, 3))
    I[0, 0] = np.sum(masses * (coords[:, 1]**2 + coords[:, 2]**2))
    I[1, 1] = np.sum(masses * (coords[:, 0]**2 + coords[:, 2]**2))
    I[2, 2] = np.sum(masses * (coords[:, 0]**2 + coords[:, 1]**2))

    I[0, 1] = I[1, 0] = -np.sum(masses * (coords[:, 0] * coords[:, 1]))
    I[0, 2] = I[2, 0] = -np.sum(masses * (coords[:, 0] * coords[:, 2]))
    I[1, 2] = I[2, 1] = -np.sum(masses * (coords[:, 1] * coords[:, 2]))
    return I


def get_dipole(scf, n=None):
    """Calculate the electric dipole moment.

    This function does not account for periodcity, it may be a good idea to center the system.

    Reference: J. Chem. Phys. 155, 224109.

    Args:
        scf: SCF object.

    Keyword Args:
        n (float): Real-space electronic density.

    Returns:
        ndarray: Electric dipole moment in e * Bohr.
    """
    atoms = scf.atoms
    if n is None:
        n = scf.n
    if scf.n is None:
        log.error('There is no density to calculate a dipole.')
        return 0

    # Diple moment: mu = \sum Z X - \int n(r) r dr
    mu = np.array([0, 0, 0], dtype=float)
    for i in range(atoms.Natoms):
        mu += atoms.Z[i] * atoms.X[i]

    dV = atoms.Omega / np.prod(atoms.s)
    for dim in range(3):
        mu[dim] -= dV * np.sum(n * atoms.r[:, dim])
    return mu


def get_ip(scf):
    """Calculate the ionization potential by calculating the negative HOMO energy.

    Reference: Physica 1, 104.

    Args:
        scf: SCF object.

    Returns:
        float: Ionization potential in Hartree.
    """
    epsilon = get_epsilon(scf, scf.W)
    # Add up spin states
    epsilon = np.sum(epsilon, axis=0)
    return -epsilon[-1]


def check_ortho(object, func, eps=1e-9):
    """Check the orthogonality condition for a set of functions.

    Args:
        object: Atoms or SCF object.
        func (ndarray): Discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Orthogonality status for the set of functions.
    """
    func = np.atleast_3d(func)
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    # It makes no sense to calculate anything for only one function
    if atoms.Nstate == 1:
        log.warning('Need at least two functions to check their orthogonality.')
        return True

    # We integrate over our cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor dV is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    dV = atoms.Omega / np.prod(atoms.s)

    ortho_bool = True
    # Check the condition for every combination
    # Orthogonality condition: \int func1^* func2 dr = 0
    for spin in range(atoms.Nspin):
        for i in range(atoms.Nstate):
            for j in range(i + 1, atoms.Nstate):
                res = dV * np.sum(func[spin, :, i].conj() * func[spin, :, j])
                tmp_bool = abs(res) < eps
                ortho_bool *= tmp_bool
                log.debug(f'Function {i} and {j}:\nValue: {res:.7f}\nOrthogonal: {tmp_bool}')
    log.info(f'Orthogonal: {ortho_bool}')
    return ortho_bool


def check_norm(object, func, eps=1e-9):
    """Check the normalization condition for a set of functions.

    Args:
        object: Atoms or SCF object.
        func (ndarray): Discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Normalization status for the set of functions.
    """
    func = np.atleast_3d(func)
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    # We integrate over our cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor dV is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    dV = atoms.Omega / np.prod(atoms.s)

    norm_bool = True
    # Check the condition for every function
    # Normality condition: \int func^* func dr = 1
    for spin in range(atoms.Nspin):
        for i in range(atoms.Nstate):
            res = dV * np.sum(func[spin, :, i].conj() * func[spin, :, i])
            tmp_bool = abs(1 - res) < eps
            norm_bool *= tmp_bool
            log.debug(f'Function {i}:\nValue: {res:.7f}\nNormalized: {tmp_bool}')
    log.info(f'Normalized: {norm_bool}')
    return norm_bool


def check_orthonorm(object, func):
    """Check the orthonormality conditions for a set of functions.

    Args:
        object: Atoms or SCF object.
        func (ndarray): Discretized set of functions.

    Returns:
        bool: Orthonormality status for the set of functions.
    """
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    ortho_bool = check_ortho(atoms, func)
    norm_bool = check_norm(atoms, func)
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
        ndarray: Real space Thomas-Fermi kinetic energy density.
    """
    atoms = scf.atoms
    # Use the definition with a division by two
    tautf = 3 / 10 * (atoms.Nspin * 3 * np.pi**2)**(2 / 3) * scf.n_spin**(5 / 3)

    log.debug(f'Calculated Ekin:  {scf.energies.Ekin:.6f} Eh')
    log.debug(f'Integrated tautf: {np.sum(tautf) * atoms.Omega / np.prod(atoms.s):.6f} Eh')
    return tautf


def get_tauw(scf):
    """Calculate the von Weizsaecker kinetic energy densities per spin.

    Reference: Z. Phys. 96, 431.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Real space von Weizsaecker kinetic energy density.
    """
    atoms = scf.atoms
    if scf.dn_spin is None:
        dn_spin = get_grad_field(atoms, scf.n_spin)
    else:
        dn_spin = scf.dn_spin
    dn2 = norm(dn_spin, axis=2)**2
    # Use the definition with a division by two
    tauw = dn2 / (8 * scf.n_spin)

    # For one- and two-electron systems the integrated KED has to be the same as the calculated KE
    log.debug(f'Calculated Ekin: {scf.energies.Ekin:.6f} Eh')
    log.debug(f'Integrated tauw: {np.sum(tauw) * atoms.Omega / np.prod(atoms.s):.6f} Eh')
    return tauw


def get_elf(scf):
    """Calculate the electron localization function.

    Reference: J. Chem. Phys. 92, 5397.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Real space electron localization function.
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
        ndarray: Real space educed density gradient.
    """
    atoms = scf.atoms
    if scf.dn_spin is None:
        dn_spin = get_grad_field(atoms, scf.n_spin)
    else:
        dn_spin = scf.dn_spin
    norm_dn = norm(np.sum(dn_spin, axis=0), axis=1)

    kf = (3 * np.pi**2 * scf.n)**(1 / 3)
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
    if atoms.Nspin == 1:
        return 0

    rhoXr = scf.n_spin[0] - scf.n_spin[1]
    rhoXr[rhoXr < 0] = 0
    rhoX = np.sum(rhoXr) * atoms.Omega / np.prod(atoms.s)
    SX = 0.5 * (np.sum(scf.n_spin[0]) - np.sum(scf.n_spin[1])) * atoms.Omega / np.prod(atoms.s)
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
