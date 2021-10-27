#!/usr/bin/env python3
'''
Atoms object definitions.
'''
from re import sub

import numpy as np
from numpy.linalg import det, eig, inv
from scipy.fft import next_fast_len

from .energies import Energy
from .gth import init_gth_loc, init_gth_nonloc, read_gth
from .operators import I, Idag, J, Jdag, K, L, Linv, O, T
from .potentials import init_pot
from .tools import center_of_mass, cutoff2gridspacing, inertia_tensor


class Atoms:
    '''Atoms object that holds all necessary calculation parameters.

    Args:
        atom : str or list of str
            Atom symbols.
            Examples: 'CH4'; ['C', 'H', 'H', 'H', 'H']

        X : list or array of floats
            Atom positions.
            Examples: [0, 0, 0]; array([0, 0, 0]); [[0, 0, 0], [1, 1, 1]];
            array([[[0, 0, 0], [1, 1, 1]]])

    Kwargs:
        a : float list or array of floats
            Cell size or vacuum size. A cuboidic box with the same side lengths will be created.
            Default: 20 Bohr (ca. 10.5 Angstrom).

        ecut : float or None
            Cut-off energy. None will disable the G-Vector truncation (needs a separate S).
            Default: 20 Hartree (ca. 544 eV).

        Z : int or list or array of ints or None
            Valence charge per atom. The charges should not differ for same species.
            None will use valence charges from GTH pseudopotentials. The same charge for every atom
            will be assumed for single integers.
            Example: 1; [4, 1, 1, 1, 1]
            Default: None

        S : int or list or array of ints
            Real-space sampling of the cell/vacuum. None will make the sampling dependent of a and
            ecut.
            Example: 30; [30, 40, 50]; array([30, 40, 50])
            Default: None

        f : float or list or array of floats
            Occupation numbers per state. None will assume occupations of 2. The last state will
            be adjusted if the sum of f is not equal to the sum of Z.
            Example: 2; [2, 2/3, 2/3, 2/3]; array([2, 2/3, 2/3, 2/3])
            Default: None

        Ns : int
            Number of states. None will get the number of states from f or assume occupations of 2
            and divide the sum of Z by it.
            Default: None

        verbose : int
            Level of output. Larger numbers mean more output.
            Default: 3

        pot : str
            Type of pseudopotential (case insensitive).
            Example: 'GTH'; 'harmonic'; 'Coulomb'; 'Ge'
            Default: 'gth'

        center : bool
            Center the system inside the box by its geometric center of mass and rotate it such
            that its geometric moment of inertia aligns with the coordinate axes.
            Default: False

        exc : str
            Exchange-correlation functional description (case insensitive), separated by a comma.
            Example: 'lda,vwn'; 'lda,pw'; 'lda,'; ',vwn';0 ',pw'; ','
            Default: 'lda,vwn'

        spinpol : bool
            Spin-polarized calculation.
            Default: False

        cutcoul : float
            Radius of the spherical truncation of the Coulomb potential. None will set the
            theoretical minimum of sqrt(3)*a.
            Default: None
    '''

    def __init__(self, atom, X, a=20, ecut=20, Z=None, S=None, f=None, Ns=None, verbose=3,
                 pot='gth', center=False, exc='lda,vwn', spinpol=False, cutcoul=None):
        self.atom = atom          # Atom symbols
        self.X = X                # Atom positions
        self.a = a                # Cell/Vacuum size
        self.ecut = ecut          # Cut-off energy
        self.Z = Z                # Valence charges
        self.S = S                # Cell sampling
        self.f = f                # Occupation numbers
        self.Ns = Ns              # Number of states
        self.pot = pot            # Used pseudopotential
        self.verbose = verbose    # Output control
        self.center = center      # Center molecule in cell
        self.exc = exc            # Exchange-correlation functional
        self.spinpol = spinpol    # Bool for spin polarized calculations
        self.cutcoul = cutcoul    # Cut-off radius for a spherical coulomb truncation

        # Parameters that will be built out of the inputs
        self.Natoms = None    # Number of atoms
        self.R = None         # Unit cell
        self.CellVol = None   # Unit cell volume
        self.r = None         # Sample points in unit cell
        self.G = None         # G-vectors
        self.G2 = None        # Squared magnitudes of G-vectors
        self.Sf = None        # Structure factor
        self.active = None    # Mask for active G-vectors
        self.Gc = None        # Truncated G-vectors
        self.G2c = None       # Truncated squared magnitudes of G-vectors
        self.GTH = {}         # Dictionary of GTH parameters per atom species
        self.Vloc = None      # Local pseudopotential contribution
        self.NbetaNL = 0      # Number of projector functions for the non-local gth potential
        self.prj2beta = None  # Index matrix to map to the correct projector function
        self.betaNL = None    # Atomic-centered projector functions

        # Parameters after SCF calculations
        self.W = None             # Basis functions
        self.n = None             # Electronic density
        self.energies = Energy()  # Energy object that holds energy contributions

        self.update()

    def update(self):
        '''Check inputs and update them if none are given.'''
        # If a string is given for atom symbols convert them to a list of strings
        if isinstance(self.atom, str):
            # Insert a whitespace before every capital letter, these can appear once or none at all
            # Or insert before digits, these can appear at least once
            self.atom = sub(r'([A-Z?]|\d+)', r' \1', self.atom).split()
            atom = []
            for ia in self.atom:
                if ia.isdigit():
                    # if ia is a integer, append the previous atom ia-1 times
                    atom += [atom[-1]] * (int(ia) - 1)
                else:
                    # If ia is a string, add it to the results list
                    atom += [ia]
            self.atom = atom

        self.Natoms = len(self.atom)

        # We need atom positions as a two-dimensional array
        self.X = np.asarray(self.X)
        if self.X.ndim == 1:
            self.X = np.array([self.X])

        # If only one charge is given, assume it is the charge for every atom
        if isinstance(self.Z, (int, np.integer)):
            self.Z = [self.Z] * self.Natoms
        if isinstance(self.Z, (list, tuple)):
            self.Z = np.asarray(self.Z)

        if isinstance(self.a, (int, np.integer, float, np.floating)):
            self.a = self.a * np.ones(3)
        if isinstance(self.a, (list, tuple)):
            self.a = np.asarray(self.a)

        # Make sampling dependent of ecut if no sampling is given
        if self.S is None:
            try:
                S = np.int_(self.a / cutoff2gridspacing(self.ecut))
            except TypeError:
                print('ERROR: No ecut provided, please enter a valid S.')
            # Multiply by two and add one to match PWDFT
            S = 2 * S + 1
            # Calculate a fast length to optimize the FFT calculations
            # See https://github.com/scipy/scipy/blob/master/scipy/fft/_helper.py
            for i in range(len(S)):
                S[i] = next_fast_len(S[i])
            self.S = S
        # Choose the same sampling for every direction if an integer is given
        if isinstance(self.S, (int, np.integer)):
            self.S = self.S * np.ones(3, dtype=int)
        if isinstance(self.S, (list, tuple)):
            self.S = np.asarray(self.S)

        # Lower the potential string
        self.pot = self.pot.lower()

        # If the cut-off radius is zero, set it to the theoretical minimal value
        if self.cutcoul == 0:
            self.cutcoul = np.sqrt(3) * self.a[0]

        # Center molecule by its geometric center of mass in the unit cell
        # Also rotate it such that the geometric inertia tensor will be diagonal
        if self.center:
            # Rotate the system
            X = self.X
            I = inertia_tensor(X)
            _, eigvecs = eig(I)
            X = np.dot(inv(eigvecs), X.T).T

            # Shift to center of the box
            com = center_of_mass(X)
            self.X = X - (com - self.a / 2)

        # Lower the exchange-correlation string
        self.exc = self.exc.lower()

        # Build a cuboidic unit cell and calculate its volume
        R = self.a * np.eye(3)
        self.R = R
        self.CellVol = np.abs(det(R))

        # Build index matrix M
        ms = np.arange(np.prod(self.S))
        m1 = ms % self.S[0]
        m2 = np.floor(ms / self.S[0]) % self.S[1]
        m3 = np.floor(ms / (self.S[0] * self.S[1])) % self.S[2]
        M = np.array([m1, m2, m3]).T

        # Build index matrix N
        n1 = m1 - (m1 > self.S[0] / 2) * self.S[0]
        n2 = m2 - (m2 > self.S[1] / 2) * self.S[1]
        n3 = m3 - (m3 > self.S[2] / 2) * self.S[2]
        N = np.array([n1, n2, n3]).T

        # Build sampling points
        r = M @ inv(np.diag(self.S)) @ R.T
        self.r = r

        # Build G-vectors
        G = 2 * np.pi * N @ inv(R)
        self.G = G

        # Calculate squared-magnitudes of G-vectors
        G2 = np.sum(G**2, axis=1)
        self.G2 = G2

        # Calculate the structure factor per atom
        Sf = np.exp(1j * G @ self.X.conj().T).T
        self.Sf = Sf

        # Restrict the G and G2
        if self.ecut is not None:
            active = np.nonzero(G2 <= 2 * self.ecut)
        else:
            active = np.nonzero(G2 >= 0)  # Trivial condition to produce the right shape
        self.active = active
        self.Gc = G[active]
        self.G2c = G2[active]

        # Update the potentials
        if self.pot == 'gth':
            if self.Z is not None:
                for ia in range(self.Natoms):
                    self.GTH[self.atom[ia]] = read_gth(self.atom[ia], self.Z[ia])
            else:
                # If no charges are given, use the ones provided by the pseudopotential
                Z = []
                for ia in range(self.Natoms):
                    self.GTH[self.atom[ia]] = read_gth(self.atom[ia])
                    Z.append(self.GTH[self.atom[ia]]['Zval'])
                self.Z = np.asarray(Z)
            # Setup potentials
            self.Vloc = init_gth_loc(self)
            self.NbetaNL, self.prj2beta, self.betaNL = init_gth_nonloc(self)
        else:
            self.Vloc = init_pot(self)

        # Check occupations and number of states together
        if isinstance(self.f, (list, tuple)):
            self.f = np.asarray(self.f)
        # If no states are provided, use the length of the occupations
        if isinstance(self.f, np.ndarray) and self.Ns is None:
            self.Ns = len(self.f)
        # If one occupation and the number of states is given, use it for every state
        if isinstance(self.f, (int, np.integer, float, np.floating)) and self.Ns is not None:
            self.f = self.f * np.ones(self.Ns)
        # If no occupation and the number of states is given, assume 2
        if self.f is None and self.Ns is not None:
            self.f = 2 * np.ones(self.Ns)
        # If number of states is None and occupations is a Number or None, we are in trouble
        if self.Ns is None:
            # If no occupations is given, assume 2
            if self.f is None:
                self.f = 2
            # Assume the number of states by dividing the total valence charge by the occupation
            Ztot = np.sum(self.Z)
            self.Ns = int(np.ceil(Ztot / self.f))
        # If the sum of valence charges is not divisible by occupation, change the last occupation
        if isinstance(self.f, (int, np.integer, float, np.floating)):
            mod = np.sum(self.Z) % self.f
            self.f = self.f * np.ones(self.Ns)
            if mod != 0:
                self.f[-1] = mod
        return

    def __repr__(self):
        atom = self.atom
        Natoms = self.Natoms
        X = self.X
        Z = self.Z

        out = 'Atom\tCharge\tPosition'
        for i in range(Natoms):
            out = f'{out}\n{atom[i]}\t{Z[i]}\t{X[i][0]:10.5f}  {X[i][1]:10.5f}  {X[i][2]:10.5f}'
        return out

    def O(self, inp):
        '''Overlap operator :func:`~eminus.operators.O`.'''
        return O(self, inp)

    def L(self, inp):
        '''Laplacian operator :func:`~eminus.operators.L`.'''
        return L(self, inp)

    def Linv(self, inp):
        '''Inverse Laplacian operator :func:`~eminus.operators.Linv`.'''
        return Linv(self, inp)

    def K(self, inp):
        '''Preconditioning operator :func:`~eminus.operators.K`.'''
        return K(self, inp)

    def I(self, inp):
        '''Transformation from k- to real-space :func:`~eminus.operators.I`.'''
        return I(self, inp)

    def J(self, inp, full=True):
        '''Transformation from real- to k-space :func:`~eminus.operators.J`.'''
        return J(self, inp, full)

    def Idag(self, inp):
        '''Conj transformation from k- to real-space :func:`~eminus.operators.Idag`.'''
        return Idag(self, inp)

    def Jdag(self, inp):
        '''Conj transformation from real- to k-space :func:`~eminus.operators.Jdag`.'''
        return Jdag(self, inp)

    def T(self, inp, dr):
        '''Translation operator :func:`~eminus.operators.T`.'''
        return T(self, inp, dr)
