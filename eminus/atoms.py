#!/usr/bin/env python3
"""Atoms class definition."""
import numbers

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import det, eigh, inv, norm

from . import config, operators
from .logger import create_logger, get_level, log
from .occupations import Occupations
from .tools import center_of_mass, cutoff2gridspacing, inertia_tensor
from .utils import atom2charge, molecule2list


class Atoms:
    """Atoms object that holds all system and cell parameters.

    Args:
        atom (str | list | tuple): Atom symbols.

            A string can be given, e.g., with :code:`CH4` that will be parsed to
            :code:`['C', 'H', 'H', 'H', 'H']`. When calculating atoms one can directly provide the
            charge, e.g., with :code:`Li-q3`.
        pos (list | tuple | ndarray): Atom positions.

    Keyword Args:
        ecut (float | None): Cut-off energy.

            Defaults to 30 Eh (ca. 816 eV).
        a (float | list | tuple | ndarray): Cell size or vacuum size.

            Floats will create a cubic unit cell. Defaults to a 20 a0 (ca. 10.5 A) cubic cell.
            Scaled lattice vectors can be given to build a custom cell.
        spin (int | None): Number of unpaired electrons.

            This is the difference between the number of up and down electrons.
        charge (int): Charge of the system.
        unrestricted (bool | None): Handling of spin.

            :code:`False` for restricted, :code:`True` for unrestricted, and :code:`None` for
            automatic detection.
        center (bool | str): Center the system inside the cell.

            Aligns the geometric center of mass with the center of the call and rotates the system,
            such that its geometric moment of inertia aligns with the coordinate axes. Can be one of
            bool, 'shift', and 'rotate'.
        verbose (int | str | None): Level of output.

            Can be one of 'critical', 'error', 'warning', 'info' (default), or 'debug'. An integer
            value can be used as well, where larger numbers mean more output, starting from 0.
            None will use the global logger verbosity value.
    """
    def __init__(self, atom, pos, ecut=30, a=20, spin=None, charge=0, unrestricted=None,
                 center=False, verbose=None):
        """Initialize the Atoms object."""
        # Set the input parameters (the ordering is important)
        self.log = create_logger(self)    #: Logger object.
        self.verbose = verbose            #: Verbosity level.
        self.occ = Occupations()          #: Occupations object.
        self.atom = atom                  #: Atom symbols.
        self.pos = pos                    #: Atom positions.
        self.a = a                        #: Cell/Vacuum size.
        self.ecut = ecut                  #: Cut-off energy.
        self.center = center              #: Enables centering the system in the cell.
        self.charge = charge              #: System charge.
        self.spin = spin                  #: Number of unpaired electrons.
        self.unrestricted = unrestricted  #: Enables unrestricted spin handling.

        # Initialize other attributes
        self.occ.fill()                   #: Fill states from the given input.
        self.is_built = False             #: Determines the Atoms object build status.

    # ### Class properties ###

    @property
    def atom(self):
        """Atom symbols."""
        return self._atom

    @atom.setter
    def atom(self, value):
        # Quick option to set the charge for single atoms
        if isinstance(value, str) and '-q' in value:
            atom, Z = value.split('-q')
            self._atom = [atom]
            self._Natoms = 1
            self.Z = int(Z)
        else:
            # If a string, i.e., chemical formula is given convert it to a list of chemical symbols
            if isinstance(value, str):
                self._atom = molecule2list(value)
            else:
                self._atom = value
            # Get the number of atoms and determine the charges
            self._Natoms = len(self._atom)
            self.Z = None

    @property
    def pos(self):
        """Atom positions."""
        return self._pos

    @pos.setter
    def pos(self, value):
        # We need atom positions as a two-dimensional array
        self._pos = np.atleast_2d(value)
        if self.Natoms != len(self._pos):
            raise ValueError(f'Mismatch between number of atoms ({self.Natoms}) and number of '
                             f'coordinates ({len(self._pos)}).')
        # The structure factor changes when changing pos
        self.is_built = False

    @property
    def ecut(self):
        """Cut-off energy."""
        return self._ecut

    @ecut.setter
    def ecut(self, value):
        self._ecut = value
        # Caclulate the sampling from the cut-off energy
        s = np.int64(norm(self.a, axis=0) / cutoff2gridspacing(value))
        # Multiply by two and add one to match PWDFT.jl
        s = 2 * s + 1
        # Calculate a fast length to optimize the FFT calculations
        self.s = [next_fast_len(i) for i in s]
        # The cell discretization changes when changing s or ecut
        self.is_built = False

    @property
    def a(self):
        """Cell/Vacuum size."""
        return self._a

    @a.setter
    def a(self, value):
        # Build a cubic cell if a number or 1d-array is given
        if np.asarray(value).ndim <= 1:
            self._a = value * np.eye(3)
        # Otherwise scaled cell vectors are given
        else:
            self._a = np.asarray(value)
        # Update ecut and s if it has been set before
        if hasattr(self, 'ecut'):
            self.ecut = self.ecut
        # Calculate the unit cell volume
        self._Omega = abs(det(self._a))
        # The cell changes when changing a
        self.is_built = False

    @property
    def spin(self):
        """Number of unpaired electrons."""
        return self.occ.spin

    @spin.setter
    def spin(self, value):
        self.occ.spin = value

    @property
    def charge(self):
        """System charge."""
        return self.occ.charge

    @charge.setter
    def charge(self, value):
        self.occ.charge = value

    @property
    def unrestricted(self):
        """Enables unrestricted spin handling."""
        return self.occ.Nspin == 2

    @unrestricted.setter
    def unrestricted(self, value):
        if value is None:
            self.occ.Nspin = value
        else:
            self.occ.Nspin = value + 1

    @property
    def center(self):
        """Enables centering the system in the cell."""
        return self._center

    @center.setter
    def center(self, value):
        if isinstance(value, str):
            self._center = value.lower()
            if self._center not in ('rotate', 'shift', 'recentered'):
                log.error(f'{self._center} is not a recognized center method.')
        else:
            self._center = value
        # Do nothing when recentering
        if self._center == 'recentered':
            return
        # Center system such that the geometric inertia tensor will be diagonal
        # Rotate before shifting!
        if self._center is True or self._center == 'rotate':
            I = inertia_tensor(self.pos)
            _, eigvecs = eigh(I)
            self.pos = (inv(eigvecs) @ self.pos.T).T
        # Shift system such that its geometric center of mass is in the center of the cell
        if self._center is True or self._center == 'shift':
            com = center_of_mass(self.pos)
            self.pos = self.pos - (com - np.sum(self.a, axis=0) / 2)
        # The structure factor changes when changing pos
        self.is_built = False

    @property
    def verbose(self):
        """Verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        # If no verbosity is given use the global verbosity level
        if value is None:
            value = log.verbose
        self._verbose = get_level(value)
        self.log.verbose = self._verbose

    # ### Class properties with a setter outside of the init method ###

    @property
    def f(self):
        """Occupation numbers per state."""
        return self.occ.f

    @f.setter
    def f(self, value):
        # Pass through to the Occupations object
        self.occ.f = value

    @property
    def s(self):
        """Real-space sampling of the cell."""
        return self._s

    @s.setter
    def s(self, value):
        # Choose the same sampling for every direction if an integer is given
        if isinstance(value, numbers.Integral):
            value = value * np.ones(3, dtype=int)
        self._s = np.asarray(value)
        self._Ns = int(np.prod(self._s))
        # The cell discretization changes when changing s
        self.is_built = False

    @property
    def Z(self):
        """Valence charge per atom."""
        return self._Z

    @Z.setter
    def Z(self, value):
        # Assume same charges for all atoms if an integer is given
        if isinstance(value, numbers.Integral):
            value = value * np.ones(self.Natoms, dtype=int)
        elif isinstance(value, dict):
            value = [value[ia] for ia in self.atom]
        # Get the valence charges from the GTH files
        elif value is None or isinstance(value, str):
            value = atom2charge(self.atom, value)
        self._Z = np.asarray(value)
        if self.Natoms != len(self._Z):
            raise ValueError(f'Mismatch between number of atoms ({self.Natoms}) and number of '
                             f'charges ({len(self._Z)}).')
        # Get the number of calculated electrons and pass it to occ
        self.occ.Nelec = np.sum(self._Z) - self.charge

    # ### Read-only properties ###

    @property
    def Natoms(self):
        """Number of atoms."""
        return self._Natoms

    @property
    def Ns(self):
        """Number of real-space grid points."""
        return self._Ns

    @property
    def Omega(self):
        """Unit cell volume."""
        return self._Omega

    @property
    def r(self):
        """Real-space sampling points."""
        return self._r

    @property
    def active(self):
        """Mask for active G-vectors."""
        return self._active

    @property
    def G(self):
        """G-vectors."""
        return self._G

    @property
    def G2(self):
        """Squared magnitudes of G-vectors."""
        return self._G2

    @property
    def G2c(self):
        """Truncated squared magnitudes of G-vectors."""
        return self._G2c

    @property
    def Gk2(self):
        """Squared magnitudes of G+k-vectors.."""
        return self._Gk2

    @property
    def Gk2c(self):
        """Truncated squared magnitudes of G+k-vectors."""
        return self._Gk2c

    @property
    def Sf(self):
        """Structure factor per atom."""
        return self._Sf

    @property
    def dV(self):
        """Volume element to multiply when integrating field properties."""
        return self.Omega / self._Ns

    @property
    def _atoms(self):
        """Return the Atoms object itself."""
        # This way we can access the object from Atoms and SCF classes with the same code
        return self

    # ### Class methods ###

    def build(self):
        """Build all parameters of the Atoms object."""
        self._set_operators()
        self._sample_unit_cell()
        self.occ.fill()
        self.is_built = True
        return self

    kernel = build

    def recenter(self, center=None):
        """Recenter the system inside the cell.

        Keyword Args:
            center (float | list | tuple | ndarray | None): Point to center the system around.
        """
        com = center_of_mass(self.pos)
        if center is None:
            self.pos = self.pos - (com - np.sum(self.a, axis=0) / 2)
        else:
            center = np.asarray(center)
            self.pos = self.pos - (com - center)
        # Recalculate the structure factor since it depends on the atom positions
        self._Sf = np.exp(1j * self.G @ self.pos.T).T
        self._center = 'recentered'
        return self

    def clear(self):
        """Initialize or clear parameters that will be built out of the inputs."""
        self._r = None          # Sample points in cell
        self._G = None          # G-vectors
        self._G2 = None         # Squared magnitudes of G-vectors
        self._active = None     # Mask for active G-vectors
        self._G2c = None        # Truncated squared magnitudes of G-vectors
        self._Sf = None         # Structure factor
        self.is_built = False   # Flag to determine if the object was built or not
        return self

    def _get_index_matrices(self):
        """Build index matrices M and N to build the real and reciprocal space samplings.

        The matrices are using C ordering (the last index is the fastest).

        Returns:
            tuple[ndarray, ndarray]: Index matrices.
        """
        # Build index matrix M
        ms = np.arange(self._Ns)
        m1 = np.floor(ms / (self.s[2] * self.s[1])) % self.s[0]
        m2 = np.floor(ms / self.s[2]) % self.s[1]
        m3 = ms % self.s[2]
        M = np.column_stack((m1, m2, m3))
        # Build index matrix N
        n1 = m1 - (m1 > self.s[0] / 2) * self.s[0]
        n2 = m2 - (m2 > self.s[1] / 2) * self.s[1]
        n3 = m3 - (m3 > self.s[2] / 2) * self.s[2]
        N = np.column_stack((n1, n2, n3))
        return M, N

    def _sample_unit_cell(self):
        """Build the real-space sampling and all G-vector parameters."""
        # Calculate index matrices
        M, N = self._get_index_matrices()
        # Build the real-space sampling
        self._r = M @ inv(np.diag(self.s)) @ self.a.T
        # Build G-vectors
        self._G = 2 * np.pi * N @ inv(self.a)
        # Calculate squared magnitudes of G-vectors
        self._G2 = norm(self.G, axis=1)**2
        # Calculate the G2 restriction
        self._active = [np.nonzero(2 * self.ecut >= norm(self.G + self.k[ik], axis=1)**2) for ik in range(len(self.wk))]
        self._G2c = self.G2[np.nonzero(2 * self.ecut >= self._G2)]
        # Calculate G+k-vectors
        self._Gk2 = np.asarray([norm(self.G + self.k[ik], axis=1)**2 for ik in range(len(self.wk))])
        self._Gk2c = [self.Gk2[ik][self._active[ik]] for ik in range(len(self.wk))]
        # Calculate the structure factor per atom
        self._Sf = np.exp(1j * self.G @ self.pos.T).T

    def _base_operator(*args, **kwargs):
        """See :mod:`~eminus.operators`."""
        # Base operator method to show an error for unbuilt Atoms objects
        log.error('Build the Atoms object with "atoms.build()" to call this operator.')

    def _set_operators(self):
        """Set operators of an Atoms class instance at runtime."""
        for op in ('O', 'L', 'Linv', 'K', 'T'):
            setattr(type(self), op, getattr(operators, op))
        fft_operators = ('I', 'J', 'Idag', 'Jdag')
        # Use the Torch operators if desired, or the default ones otherwise
        if config.use_torch:
            from .extras import torch
            for op in fft_operators:
                setattr(type(self), op, getattr(torch, op))
        else:
            for op in fft_operators:
                setattr(type(self), op, getattr(operators, op))

    O = _base_operator
    L = _base_operator
    Linv = _base_operator
    K = _base_operator
    T = _base_operator
    I = _base_operator
    J = _base_operator
    Idag = _base_operator
    Jdag = _base_operator

    def __repr__(self):
        """Print the parameters stored in the Atoms object."""
        out = 'Atom  Valence  Position'
        for i in range(self.Natoms):
            out += f'\n{self.atom[i]:>3}   {self.Z[i]:>6}   ' \
                   f'{self.pos[i, 0]:10.5f}  {self.pos[i, 1]:10.5f}  {self.pos[i, 2]:10.5f}'
        return out
