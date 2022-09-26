#!/usr/bin/env python3
'''Atoms class definition.'''
import re

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import det, eig, inv, norm

from .io import read_gth
from .logger import create_logger, get_level, log
from .operators import I, Idag, J, Jdag, K, L, Linv, O, T
from .tools import center_of_mass, cutoff2gridspacing, inertia_tensor


class Atoms:
    '''Atoms object that holds all system and cell parameters.

    Args:
        atom (str | list | tuple): Atom symbols.

            Examples: 'CH4'; ['C', 'H', 'H', 'H', 'H']; ('C', 'H', 'H', 'H', 'H')
        X (list | tuple | ndarray): Atom positions.

            Examples: (0, 0, 0); array([0, 0, 0]); [[0, 0, 0], [1, 1, 1]];

    Keyword Args:
        a (float | list | tuple | ndarray): Cell size or vacuum size.

            A cuboidal box with the same side lengths will be created.

            Examples: 10; [10, 10, 10]; (7, 8, 9),
            Default: 20 Bohr (ca. 10.5 Angstrom).
        ecut (float | None): Cut-off energy.

            None will disable the G-Vector truncation (needs a separate s).
            Default: 30 Hartree (ca. 816 eV).
        Z (int | list | tuple | ndarray | None): Valence charge per atom.

            The charges should not differ for same species. None will use valence charges from GTH
            files. The same charge for every atom will be assumed for single integers.

            Example: 1; [4, 1, 1, 1, 1],
            Default: None
        s (int | list | tuple | ndarray | None): Real-space sampling of the cell.

            None will make the sampling dependent on a and ecut.

            Example: 30; [30, 40, 50]; array([30, 40, 50]),
            Default: None
        center (bool | str): Center the system inside the cell (case insensitive).

            Align the geometric center of mass with the center of the call and rotate the system,
            such that its geometric moment of inertia aligns with the coordinate axes.

            Example: True; 'shift'; 'rotate',
            Default: False
        Nspin (int | None): Number of spin states.

            1 for spin-paired, 2 for spin-polarized, None for automatic detection.
            Default: 2
        f (float | list | tuple | ndarray | None): Occupation numbers per state.

            The last state will be adjusted if the sum of f is not equal to the sum of Z.
            None will assume occupations of 2.

            Example: 2; [2, 2, 2, 2]; array([2, 2/3, 2/3, 2/3]),
            Default: None
        Nstate (int | None): Number of states.

            None will get the number of states from f or assume occupations of 2 and divide the sum
            of Z by it.

            Default: None
        verbose (int | str | None): Level of output (case insensitive).

            Can be one of 'CRITICAL', 'ERROR', 'WARNING', 'INFO', or 'DEBUG'.
            An integer value can be used as well, where larger numbers mean more output, starting
            from 0.
            None will use the default global logger value 'WARNING'.

            Default: 'info'
    '''
    def __init__(self, atom, X, a=20, ecut=30, Z=None, s=None, center=False, Nspin=2, f=None,
                 Nstate=None, verbose='info'):
        self.atom = atom      # Atom symbols
        self.X = X            # Atom positions
        self.a = a            # Cell/Vacuum size
        self.ecut = ecut      # Cut-off energy
        self.Z = Z            # Valence charges
        self.s = s            # Cell sampling
        self.center = center  # Center molecule in cell
        self.Nspin = Nspin    # Number of spin states
        self.f = f            # Occupation numbers
        self.Nstate = Nstate  # Number of states

        # Parameters that will be built out of the inputs
        self.Natoms = None  # Number of atoms
        self.R = None       # Cell
        self.Omega = None   # Cell volume
        self.r = None       # Sample points in cell
        self.G = None       # G-vectors
        self.G2 = None      # Squared magnitudes of G-vectors
        self.active = None  # Mask for active G-vectors
        self.G2c = None     # Truncated squared magnitudes of G-vectors
        self.Sf = None      # Structure factor

        # Initialize logger and update
        self.log = create_logger(self)
        if verbose is None:
            self.verbose = log.verbose
        else:
            self.verbose = verbose
        self.update()

    def update(self):
        '''Validate inputs, update them and build all necessary parameters.'''
        self._set_atom()
        self._set_charge()
        self._set_cell_size()
        self._set_positions()
        self._set_sampling()
        self._set_states(self.Nspin)
        M, N = self._get_index_matrices()
        self._set_cell(M)
        self._set_G(N)
        return

    def _set_atom(self):
        '''Validate the atom input and calculate the number of atoms.'''
        # Quick option to set the charge for single atoms
        if isinstance(self.atom, str) and '-q' in self.atom:
            atom, Z = self.atom.split('-q')
            self.atom = [atom]
            self.Z = np.asarray(list(Z), dtype=int)
        # If a string is given for atom symbols convert them to a list of strings
        if isinstance(self.atom, str):
            # Insert a whitespace before every capital letter, these can appear once or none at all
            # Or insert before digits, these can appear at least once
            self.atom = re.sub(r'([A-Z?]|\d+)', r' \1', self.atom).split()
            atom = []
            for ia in self.atom:
                if ia.isdigit():
                    # if ia is a integer, append the previous atom ia-1 times
                    atom += [atom[-1]] * (int(ia) - 1)
                else:
                    # If ia is a string, add it to the results list
                    atom += [ia]
            self.atom = atom

        # Get the number of atoms
        self.Natoms = len(self.atom)
        return

    def _set_charge(self):
        '''Validate the Z input and calculate charges if necessary.'''
        # If only one charge is given, assume it is the charge for every atom
        if isinstance(self.Z, (int, np.integer)):
            self.Z = [self.Z] * self.Natoms
        if isinstance(self.Z, (list, tuple)):
            self.Z = np.asarray(self.Z)

        # If no charge is given, use the ionic charge from the GTH files
        if self.Z is None:
            Z = []
            for ia in range(self.Natoms):
                gth_dict = read_gth(self.atom[ia])
                Z.append(gth_dict['Zion'])
            self.Z = np.asarray(Z)
        return

    def _set_cell_size(self):
        '''Validate the a input.'''
        # Do this early on, since it is needed in many functions
        if isinstance(self.a, (int, np.integer, float, np.floating)):
            self.a = self.a * np.ones(3)
        if isinstance(self.a, (list, tuple)):
            self.a = np.asarray(self.a)
        return

    def _set_positions(self):
        '''Validate the X and center input and center the system if desired.'''
        # We need atom positions as an two-dimensional array
        self.X = np.atleast_2d(self.X)
        if isinstance(self.center, str):
            self.center = self.center.lower()

        # Center system such that the geometric inertia tensor will be diagonal
        # Rotate before shifting!
        if self.center or self.center == 'rotate':
            X = self.X
            I = inertia_tensor(self.X)
            _, eigvecs = eig(I)
            self.X = (inv(eigvecs) @ self.X.T).T

        # Shift system such that its geometric center of mass is in the center of the cell
        if self.center or self.center == 'shift':
            X = self.X
            com = center_of_mass(X)
            self.X = X - (com - self.a / 2)
        return

    def _set_sampling(self):
        '''Validate the s input and calculate it if necessary.'''
        # Make sampling dependent of ecut if no sampling is given
        if self.s is None:
            try:
                s = np.int_(self.a / cutoff2gridspacing(self.ecut))
            except TypeError:
                self.log.exception('No ecut provided, please enter a valid s.')
                raise
            # Multiply by two and add one to match PWDFT
            s = 2 * s + 1
            # Calculate a fast length to optimize the FFT calculations
            # See https://github.com/scipy/scipy/blob/main/scipy/fft/_helper.py
            self.s = [next_fast_len(i) for i in s]

        # Choose the same sampling for every direction if an integer is given
        if isinstance(self.s, (int, np.integer)):
            self.s = self.s * np.ones(3, dtype=int)
        if isinstance(self.s, (list, tuple)):
            self.s = np.asarray(self.s)
        return

    def _set_states(self, Nspin):
        '''Validate the f and Nstate input and calculate the states if necessary.

        Args:
            Nspin (int | None): Number of spin states.
        '''
        # Use a spin-paired calculation for an even number of electrons
        if Nspin is None:
            if np.sum(self.Z) % 2 == 0:
                Nspin = 1
            else:
                Nspin = 2
        # Make sure Nspin is an integer
        try:
            self.Nspin = int(Nspin)
        except (TypeError, ValueError):
            self.log.exception('Nspin has to be an integer.')
            raise
        # Make sure the occupations are in an two-dimensional array
        if isinstance(self.f, (list, tuple)):
            self.f = np.atleast_2d(self.f)
        # If occupations and spin number is not equal, reset the occupations and number of states
        if isinstance(self.f, np.ndarray) and len(self.f) != self.Nspin:
            self.f = None
            self.Nstate = None

        # If no states are provided use the length of the occupations
        if isinstance(self.f, np.ndarray) and self.Nstate is None:
            self.Nstate = len(self.f[0])
        # If one occupation and the number of states is given, use it for every state
        if isinstance(self.f, (int, np.integer, float, np.floating)) and self.Nstate is not None:
            self.f = self.f * np.ones((self.Nspin, self.Nstate))
        # If no occupations and the number of states is given, assume 1 or 2
        if self.f is None and self.Nstate is not None:
            self.f = 2 / self.Nspin * np.ones((self.Nspin, self.Nstate))

        # If the number of states is None and the occupations is a number or None, we are in trouble
        if self.Nstate is None:
            # If no occupations is given, assume 1 or 2
            if self.f is None:
                f = 2 / self.Nspin
            # Assume the number of states by dividing the total valence charge by an occupation of 2
            Ztot = np.sum(self.Z)
            self.Nstate = int(np.ceil(Ztot / 2))
            self.f = f * np.ones((self.Nspin, self.Nstate))
            # Subtract the leftovers from the last spin state
            self.f[-1, -1] -= np.sum(self.Z) % 2
        return

    def _get_index_matrices(self):
        '''Build index matrices M and N to build the real and reciprocal space samplings.

        Returns:
            tuple[ndarray, ndarray]: Index matrices.
        '''
        # Build index matrix M
        ms = np.arange(np.prod(self.s))
        m1 = ms % self.s[0]
        m2 = np.floor(ms / self.s[0]) % self.s[1]
        m3 = np.floor(ms / (self.s[0] * self.s[1])) % self.s[2]
        M = np.column_stack((m1, m2, m3))

        # Build index matrix N
        n1 = m1 - (m1 > self.s[0] / 2) * self.s[0]
        n2 = m2 - (m2 > self.s[1] / 2) * self.s[1]
        n3 = m3 - (m3 > self.s[2] / 2) * self.s[2]
        N = np.column_stack((n1, n2, n3))
        return M, N

    def _set_cell(self, M):
        '''Build cell and create the respective sampling.

        Args:
            M (ndarray): Index matrix.
        '''
        # Build a cuboidal cell and calculate its volume
        R = self.a * np.eye(3)
        self.R = R
        self.Omega = np.abs(det(R))
        # Build real-space sampling points
        self.r = M @ inv(np.diag(self.s)) @ R.T
        return

    def _set_G(self, N):
        '''Build G-vectors, build squared magnitudes G2, and generate the active space.

        Args:
            N (ndarray): Index matrix.
        '''
        # Build G-vectors
        G = 2 * np.pi * N @ inv(self.R)
        self.G = G
        # Calculate squared-magnitudes of G-vectors
        G2 = norm(G, axis=1)**2
        self.G2 = G2

        # Calculate the G2 restriction
        if self.ecut is not None:
            active = np.nonzero(G2 <= 2 * self.ecut)
        else:
            active = np.nonzero(G2 >= 0)  # Trivial condition to produce the right shape
        self.active = active
        self.G2c = G2[active]

        # Calculate the structure factor per atom
        self.Sf = np.exp(1j * G @ self.X.conj().T).T
        return

    def __repr__(self):
        '''Print the parameters stored in the Atoms object.'''
        out = 'Atom\tCharge\tPosition'
        for i in range(self.Natoms):
            out = f'{out}\n{self.atom[i]}\t{self.Z[i]}\t' \
                  f'{self.X[i, 0]:10.5f}  {self.X[i, 1]:10.5f}  {self.X[i, 2]:10.5f}'
        return out

    @property
    def verbose(self):
        '''Verbosity level.'''
        return self._verbose

    @verbose.setter
    def verbose(self, level):
        '''Verbosity setter to sync the logger with the property.'''
        self._verbose = get_level(level)
        self.log.setLevel(self._verbose)
        return

    def O(self, inp):
        '''Overlap operator :func:`~eminus.operators.O`.'''
        return O(self, inp)

    def L(self, inp):
        '''Laplacian operator :func:`~eminus.operators.L`.'''
        return L(self, inp)

    def Linv(self, inp):
        '''Inverse Laplacian operator :func:`~eminus.operators.Linv`.'''
        return Linv(self, inp)

    def I(self, inp):
        '''Transformation from reciprocal to real-space :func:`~eminus.operators.I`.'''
        return I(self, inp)

    def J(self, inp, full=True):
        '''Transformation from real to reciprocal space :func:`~eminus.operators.J`.'''
        return J(self, inp, full)

    def Idag(self, inp, full=False):
        '''Conj transformation from real to reciprocal space :func:`~eminus.operators.Idag`.'''
        return Idag(self, inp, full)

    def Jdag(self, inp):
        '''Conj transformation from reciprocal to real-space :func:`~eminus.operators.Jdag`.'''
        return Jdag(self, inp)

    def K(self, inp):
        '''Preconditioning operator :func:`~eminus.operators.K`.'''
        return K(self, inp)

    def T(self, inp, dr):
        '''Translation operator :func:`~eminus.operators.T`.'''
        return T(self, inp, dr)
