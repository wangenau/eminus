#!/usr/bin/env python3
'''Various tools to check physical properties.'''
import numpy as np
from scipy.optimize import minimize_scalar

from .dft import get_epsilon
from .logger import log
from .utils import handle_spin_gracefully


def cutoff2gridspacing(E):
    '''Convert plane wave energy cut-off to a real-space grid spacing.

    Reference: Phys. Rev. B 54, 14362.

    Args:
        E (float): Energy in Hartree.

    Returns:
        float: Grid spacing in Bohr.
    '''
    return np.pi / np.sqrt(2 * E)


def gridspacing2cutoff(h):
    '''Convert real-space grid spacing to plane wave energy cut-off.

    Reference: Phys. Rev. B 54, 14362.

    Args:
        h (float): Grid spacing in Bohr.

    Returns:
        float: Cut-off in Hartree.
    '''
    return 0.5 * (np.pi / h)**2


def center_of_mass(coords, masses=None):
    '''Calculate the center of mass for a set of coordinates and masses.

    Args:
        coords (ndarray): Array of real-space coordinates.

    Keyword Args:
        masses (ndarray): Mass or weight for each coordinate.

    Returns:
        ndarray: Center of mass.
    '''
    if masses is None:
        masses = np.ones(len(coords))

    return np.sum(masses * coords.T, axis=1) / np.sum(masses)


def inertia_tensor(coords, masses=None):
    '''Calculate the inertia tensor for a set of coordinates and masses.

    Args:
        coords (ndarray): Array of real-space coordinates.

    Keyword Args:
        masses (ndarray): Mass or weight for each coordinate.

    Returns:
        ndarray: Inertia tensor.
    '''
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
    '''Calculate the electric dipole moment.

    Reference: J. Chem. Phys. 155, 224109.

    Args:
        scf: SCF object.

    Keyword Args:
        n (float): Real-space electronic density.

    Returns:
        ndarray: Electric dipole moment in e * Bohr.
    '''
    # The dipole may be extremely large. This can be because of periodic boundary conditions,
    # e.g., the density gets "smeared" to the edges if the atom sits at one edge.
    # One fix can be to center the atom/molecule inside the box.
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
    '''Calculate the ionization potential by calculating the negative HOMO energy.

    Reference: Physica 1, 104.

    Args:
        scf: SCF object.

    Returns:
        float: Ionization potential in Hartree.
    '''
    epsilon = get_epsilon(scf, scf.W)
    # Add up spin states
    epsilon = np.sum(epsilon, axis=0)
    return -epsilon[-1]


def check_ortho(object, func, eps=1e-9):
    '''Check the orthogonality condition for a set of functions.

    Args:
        object: Atoms or SCF object.
        func (ndarray): Discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Orthogonality status for the set of functions.
    '''
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
                log.debug(f'Function {i} and {j}:\n\tValue: {res:.7f}\n\tOrthogonal: {tmp_bool}')
    log.info(f'Orthogonal: {ortho_bool}')
    return ortho_bool


def check_norm(object, func, eps=1e-9):
    '''Check the normalization condition for a set of functions.

    Args:
        object: Atoms or SCF object.
        func (ndarray): Discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Normalization status for the set of functions.
    '''
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
            log.debug(f'Function {i}:\n\tValue: {res:.7f}\n\tNormalized: {tmp_bool}')
    log.info(f'Normalized: {norm_bool}')
    return norm_bool


def check_orthonorm(object, func):
    '''Check the orthonormality conditions for a set of functions.

    Args:
        object: Atoms or SCF object.
        func (ndarray): Discretized set of functions.

    Returns:
        bool: Orthonormality status for the set of functions.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    ortho_bool = check_ortho(atoms, func)
    norm_bool = check_norm(atoms, func)
    log.info(f'Orthonormal: {ortho_bool * norm_bool}')
    return ortho_bool * norm_bool


def get_isovalue(n, percent=85):
    '''Find an isovalue that contains a specified percentage of the electronic density.

    Args:
        n (float): Real-space electronic density.

    Keyword Args:
        percent (float): Amount of density that should be contained.

    Returns:
        float: Isovalue that contains the specified percentage of the density.
    '''
    def deviation(isovalue):
        n_mask = np.sum(n[n > isovalue])
        return abs(percent - (n_mask / n_ref) * 100)

    # Integrated density
    n_ref = np.sum(n)
    # Finding the isovalue is an optimization problem, minimizing the deviation above
    # The problem is bound by zero (no density) and the maximum value in n
    res = minimize_scalar(deviation, bounds=(0, np.max(n)), method='bounded')
    return res.x


@handle_spin_gracefully
def pycom(object, psirs):
    '''Calculate the orbital center of masses, e.g., from localized orbitals.

    Args:
        object: Atoms or SCF object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        bool: Center of masses.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object

    Ncom = psirs.shape[1]
    coms = np.empty((Ncom, 3))
    for i in range(Ncom):
        coms[i] = center_of_mass(atoms.r, np.real(psirs[:, i].conj() * psirs[:, i]))
    return coms
