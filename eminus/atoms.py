# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Atoms class definition."""

import numbers

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import det, eigh, inv, norm

from . import operators
from .kpoints import KPoints
from .logger import create_logger, get_level, log
from .occupations import Occupations
from .tools import center_of_mass, cutoff2gridspacing, inertia_tensor
from .utils import atom2charge, BaseObject, molecule2list


class Atoms(BaseObject):
    """Atoms object that holds all system and cell parameters.

    Args:
        atom: Atom symbols.

            A string can be given, e.g., with :code:`CH4` that will be parsed to
            :code:`["C", "H", "H", "H", "H"]`. When calculating atoms one can directly provide the
            charge, e.g., with :code:`Li-q3`.
        pos: Atom positions.

    Keyword Args:
        ecut: Cut-off energy.

            Defaults to 30 Eh (ca. 816 eV).
        a: Cell size or vacuum size.

            Floats will create a cubic unit cell. Defaults to a 20 a0 (ca. 10.5 A) cubic cell.
            Scaled lattice vectors can be given to build a custom cell.
        spin: Number of unpaired electrons.

            This is the difference between the number of up and down electrons.
        charge: Charge of the system.
        unrestricted: Handling of spin.

            :code:`False` for restricted, :code:`True` for unrestricted, and :code:`None` for
            automatic detection.
        center: Center the system inside the cell.

            Aligns the geometric center of mass with the center of the call and rotates the system,
            such that its geometric moment of inertia aligns with the coordinate axes. Can be one of
            bool, "shift", and "rotate".
        verbose: Level of output.

            Can be one of "critical", "error", "warning", "info" (default), or "debug". An integer
            value can be used as well, where larger numbers mean more output, starting from 0.
            None will use the global logger verbosity value.
    """

    def __init__(
        self,
        atom,
        pos,
        ecut=30,
        a=20,
        spin=None,
        charge=0,
        unrestricted=None,
        center=False,
        verbose=None,
    ):
        """Initialize the Atoms object."""
        # Set the input parameters (the ordering is important)
        self._log = create_logger(self)  #: Logger object.
        self.verbose = verbose  #: Verbosity level.
        self.occ = Occupations()  #: Occupations object.
        self.atom = atom  #: Atom symbols.
        self.pos = pos  #: Atom positions.
        self.a = a  #: Cell/Vacuum size.
        self.ecut = ecut  #: Cut-off energy.
        self.center = center  #: Enables centering the system in the cell.
        self.charge = charge  #: System charge.
        self.spin = spin  #: Number of unpaired electrons.
        self.unrestricted = unrestricted  #: Enables unrestricted spin handling.
        self.kpts = KPoints("sc", self.a)  #: KPoints object.

        # Initialize other attributes
        self.occ.fill()  #: Fill states from the given input.
        self.is_built = False  #: Determines the Atoms object build status.

    # ### Class properties ###

    @property
    def atom(self):
        """Atom symbols."""
        return self._atom

    @atom.setter
    def atom(self, value):
        # Quick option to set the charge for single atoms
        if isinstance(value, str) and "-q" in value:
            atom, Z = value.split("-q")
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
            msg = (
                f"Mismatch between number of atoms ({self.Natoms}) and number of "
                f"coordinates ({len(self._pos)})."
            )
            raise ValueError(msg)
        # The structure factor changes when changing pos
        self.is_built = False

    @property
    def ecut(self):
        """Cut-off energy."""
        return self._ecut

    @ecut.setter
    def ecut(self, value):
        self._ecut = value
        # Calculate the sampling from the cut-off energy
        s = np.int64(norm(self.a, axis=1) / cutoff2gridspacing(value))
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
        if hasattr(self, "ecut"):
            self.ecut = self.ecut
        # Calculate the unit cell volume
        self._Omega = abs(det(self._a))
        if hasattr(self, "kpts"):
            self.kpts.a = self._a
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
            if self._center not in {"rotate", "shift", "recentered"}:
                log.error(f"{self._center} is not a recognized center method.")
        else:
            self._center = value
        # Do nothing when recentering
        if self._center == "recentered":
            return
        # Center system such that the geometric inertia tensor will be diagonal
        # Rotate before shifting!
        if self._center is True or self._center == "rotate":
            I = inertia_tensor(self.pos)
            _, eigvecs = eigh(I)
            self.pos = (inv(eigvecs) @ self.pos.T).T
        # Shift system such that its geometric center of mass is in the center of the cell
        if self._center is True or self._center == "shift":
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
        self._log.verbose = self._verbose

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
            msg = (
                f"Mismatch between number of atoms ({self.Natoms}) and number of "
                f"charges ({len(self._Z)})."
            )
            raise ValueError(msg)
        # Get the number of calculated electrons and pass it to occ
        self.occ.Nelec = np.sum(self._Z) - self.charge
        if self.occ.Nspin and self.occ.bands < self.occ.Nelec * self.occ.Nspin // 2:
            log.warning("The number of bands is too small, reset to the minimally needed amount.")
            self.occ.bands = 0

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
        """Squared magnitudes of G+k-vectors."""
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
        self.kpts.build()
        self._sample_unit_cell()
        self.occ.wk = self.kpts.wk  # Pass the weights of k-points to the Occupations object
        self.occ.fill()
        self.is_built = True
        return self

    kernel = build

    def recenter(self, center=None):
        """Recenter the system inside the cell.

        Keyword Args:
            center: Point to center the system around.
        """
        com = center_of_mass(self.pos)
        if center is None:
            self.pos = self.pos - (com - np.sum(self.a, axis=0) / 2)
        else:
            center = np.asarray(center)
            self.pos = self.pos - (com - center)
        # Recalculate the structure factor since it depends on the atom positions
        self._Sf = np.exp(1j * self.G @ self.pos.T).T
        self._center = "recentered"
        return self

    def set_k(self, k, wk=None):
        """Interface to set custom k-points.

        Args:
            k: k-point coordinates.

        Keyword Args:
            wk: k-point weights.
        """
        self.kpts.build()
        self.kpts._k = np.atleast_2d(k)
        if wk is None:
            self.kpts._wk = np.ones(len(self.kpts._k)) / len(self.kpts._k)
        else:
            self.kpts._wk = np.asarray(wk)
        self.kpts._Nk = len(self.kpts._wk)
        self.kpts._kmesh = None
        self.occ.wk = self.kpts.wk
        self._sample_unit_cell()
        return self

    def clear(self):
        """Initialize or clear parameters that will be built out of the inputs."""
        self._r = None  # Sample points in cell
        self._G = None  # G-vectors
        self._G2 = None  # Squared magnitudes of G-vectors
        self._active = None  # Mask for active G-vectors
        self._G2c = None  # Truncated squared magnitudes of G-vectors
        self._Sf = None  # Structure factor
        self.is_built = False  # Flag to determine if the object was built or not
        return self

    def _get_index_matrices(self):
        """Build index matrices M and N to build the real and reciprocal space samplings.

        The matrices are using C ordering (the last index is the fastest).

        Returns:
            Index matrices.
        """
        # Build index matrix M
        # ms = np.arange(self._Ns)
        # m1 = np.floor(ms / (self.s[2] * self.s[1])) % self.s[0]
        # m2 = np.floor(ms / self.s[2]) % self.s[1]
        # m3 = ms % self.s[2]
        # M = np.column_stack((m1, m2, m3))
        M = np.indices(self.s).transpose((1, 2, 3, 0)).reshape((-1, 3))
        # Build index matrix N
        N = M - (self.s / 2 < M) * self.s
        return M, N

    def _sample_unit_cell(self):
        """Build the real-space sampling and all G-vector parameters."""
        # Calculate index matrices
        M, N = self._get_index_matrices()
        # Build the real-space sampling
        self._r = M @ inv(np.diag(self.s)) @ self.a
        # Build G-vectors
        self._G = 2 * np.pi * N @ inv(self.a.T)
        # Calculate squared magnitudes of G-vectors
        self._G2 = norm(self.G, axis=1) ** 2
        # Calculate the G2 restriction
        self._active = [
            np.nonzero(2 * self.ecut >= norm(self.G + self.kpts.k[ik], axis=1) ** 2)
            for ik in range(self.kpts.Nk)
        ]
        self._G2c = self.G2[np.nonzero(2 * self.ecut >= self._G2)]
        # Calculate G+k-vectors
        self._Gk2 = np.asarray(
            [norm(self.G + self.kpts.k[ik], axis=1) ** 2 for ik in range(self.kpts.Nk)]
        )
        self._Gk2c = [self.Gk2[ik][self._active[ik]] for ik in range(self.kpts.Nk)]
        # Calculate the structure factor per atom
        self._Sf = np.exp(1j * self.G @ self.pos.T).T

        # Create the grid used for the non-wave function fields and append it to the end
        self._active.append(np.nonzero(2 * self.ecut >= self._G2))
        self._Gk2 = np.vstack((self._Gk2, self._G2))
        self._Gk2c.append(self._G2c)

    O = operators.O
    L = operators.L
    Linv = operators.Linv
    K = operators.K
    T = operators.T
    I = operators.I
    J = operators.J
    Idag = operators.Idag
    Jdag = operators.Jdag

    def __repr__(self):
        """Print the parameters stored in the Atoms object."""
        out = "Atom  Valence  Position"
        for i in range(self.Natoms):
            out += (
                f"\n{self.atom[i]:>3}   {self.Z[i]:>6}   "
                f"{self.pos[i, 0]:10.5f}  {self.pos[i, 1]:10.5f}  {self.pos[i, 2]:10.5f}"
            )
        return out
