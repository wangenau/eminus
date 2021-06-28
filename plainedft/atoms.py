#!/usr/bin/env python3
'''
Atoms object that holds every relevant calculation parameters and outputs and import and export
functionality for the Atoms objects.
'''
from pickle import dump, HIGHEST_PROTOCOL, load
from re import sub
from textwrap import fill
from time import ctime

import numpy as np
from numpy.linalg import det, eig, inv
from scipy.fft import next_fast_len

from .data import symbol2number
from .gth import init_gth_loc, init_gth_nonloc, read_gth
from .operators import I, Idag, J, Jdag, K, L, Linv, O
from .potentials import init_pot
from .tools import center_of_mass, cutoff2gridspacing, inertia_tensor
from .units import ang2bohr
from .version import __version__


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
        a : float
            Cell size or vacuum size. A cubic box with the same side lengths will be created.
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
            Example: 'GTH', 'harmonic', 'Coulomb', 'Ge'
            Default: 'gth'

        center : bool
            Center the system inside the box by its geometric center of mass and rotate it such
            that its geometric moment of inertia aligns with the coordinate axes.
            Default: False

        spinpol : bool
            Spin-polarized calculation.
            Default: False

        cutcoul : float
            Radius of the spherical truncation of the Coulomb potential. None will set the
            theoretical minimum of sqrt(3)*a.
            Default: None
    '''
    def __init__(self, atom, X, a=20, ecut=20, Z=None, S=None, f=None, Ns=None, verbose=3,
                 pot='gth', center=False, spinpol=False, cutcoul=None):
        '''Initialize and update the atoms object.'''
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
        self.spinpol = spinpol    # Bool for spin polarized calculations
        self.cutcoul = cutcoul    # Cut-off radius for a spherical coulomb truncation

        # Parameters that will be built out of the inputs
        self.R = None         # Unit cell
        self.CellVol = None   # Unit cell volume
        self.M = None         # Index matrix
        self.N = None         # Index matrix
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
        self.W = None       # Basis functions
        self.psi = None     # States
        self.estate = None  # Energy per state
        self.n = None       # Electronic density
        self.eewald = None  # Ewald energy
        self.etot = None    # Total energy

    def update(self):
        '''Check inputs and update if no inputs are given.'''
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

        # We need atom positions as a two-dimensional array
        self.X = np.asarray(self.X)
        if self.X.ndim == 1:
            self.X = np.array([self.X])

        # If only one charge is given, assume it is the charge for every atom
        if isinstance(self.Z, int):
            self.Z = [self.Z] * len(self.X)
        if isinstance(self.Z, list):
            self.Z = np.asarray(self.Z)

        # Make sampling dependent of ecut if no sampling is given
        if self.S is None:
            try:
                S = int(self.a / cutoff2gridspacing(self.ecut))
            except TypeError:
                print('ERROR: No ecut provided, please enter a valid S.')
            # Multiply by two and add one to match PWDFT
            S = 2 * S + 1
            # Calculate a fast length to optimize the FFT calculations
            # See https://github.com/scipy/scipy/blob/master/scipy/fft/_helper.py
            self.S = next_fast_len(S)
        # Choose the same sampling for every direction if an integer is given
        if isinstance(self.S, int):
            self.S = self.S * np.ones(3, dtype=int)
        if isinstance(self.S, list):
            self.S = np.asarray(self.S)

        # Lower the potential string
        self.pot = self.pot.lower()

        # If the cut-off radius is zero, set it to the theoretical minimal value
        if self.cutcoul == 0:
            self.cutcoul = np.sqrt(3) * self.a

        # Center molecule by its geometric center of mass in the unit cell
        # Also rotate it such that the geometric inertia tensor will be diagonal
        if self.center:
            # Shift to center of the box
            X = self.X
            com = center_of_mass(X)
            X = X - (com - self.a / 2)

            # Rotate the system
            I = inertia_tensor(X)
            _, eigvecs = eig(I)
            self.X = np.dot(inv(eigvecs), X.T).T

        # Build a cubic unit cell and calculate its volume
        R = self.a * np.eye(3)
        self.R = R
        self.CellVol = np.abs(det(R))

        # Build index matrix M
        ms = np.arange(0, np.prod(self.S))
        m1 = ms % self.S[0]
        m2 = np.floor(ms / self.S[0]) % self.S[1]
        m3 = np.floor(ms / (self.S[0] * self.S[1])) % self.S[2]
        M = np.array([m1, m2, m3]).T
        self.M = M

        # Build index matrix N
        n1 = m1 - (m1 > self.S[0] / 2) * self.S[0]
        n2 = m2 - (m2 > self.S[1] / 2) * self.S[1]
        n3 = m3 - (m3 > self.S[2] / 2) * self.S[2]
        N = np.array([n1, n2, n3]).T
        self.N = N

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
        Sf = np.exp(-1j * G @ self.X.conj().T).T
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
                for ia in range(len(self.X)):
                    self.GTH[self.atom[ia]] = read_gth(self.atom[ia], self.Z[ia])
            else:
                # If no charges are given, use the ones provided by the pseudopotential
                Z = []
                for ia in range(len(self.X)):
                    self.GTH[self.atom[ia]] = read_gth(self.atom[ia])
                    Z.append(self.GTH[self.atom[ia]]['Zval'])
                self.Z = np.asarray(Z)
            # Setup potentials
            self.Vloc = init_gth_loc(self)
            self.NbetaNL, self.prj2beta, self.betaNL = init_gth_nonloc(self)
        else:
            self.Vloc = init_pot(self)

        # Check occupations and number of states together
        if isinstance(self.f, list):
            self.f = np.asarray(self.f)
        # If no states are provided, use the length of the occupations
        if isinstance(self.f, np.ndarray) and self.Ns is None:
            self.Ns = len(self.f)
        # If one occupation and the number of states is given, use it for every state
        if isinstance(self.f, (int, float)) and self.Ns is not None:
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
        if isinstance(self.f, (int, float)):
            mod = np.sum(self.Z) % self.f
            self.f = self.f * np.ones(self.Ns)
            if mod != 0:
                self.f[-1] = mod

    def __repr__(self):
        '''Display informations when printing the Atoms object.'''
        atom = self.atom
        X = self.X
        Z = self.Z

        out = 'Atom\tCharge\tPosition'
        for i in range(len(X)):
            out = f'{out}\n{atom[i]}\t{Z[i]}\t{X[i][0]:10.5f}  {X[i][1]:10.5f}  {X[i][2]:10.5f}'
        return out

    def O(self, inp):
        '''Overlap operator.'''
        return O(self, inp)

    def L(self, inp):
        '''Laplacian operator.'''
        return L(self, inp)

    def Linv(self, inp):
        '''Inverse Laplacian operator.'''
        return Linv(self, inp)

    def K(self, inp):
        '''Precondition by applying 1/(1+G2) to the input.'''
        return K(self, inp)

    def I(self, inp):
        '''Backwards transformation from reciprocal space to real-space.'''
        return I(self, inp)

    def J(self, inp):
        '''Forward transformation from real-space to reciprocal space.'''
        return J(self, inp)

    def Idag(self, inp):
        '''Conjugated backwards transformation from reciprocal space to real-space.'''
        return Idag(self, inp)

    def Jdag(self, inp):
        '''Conjugated forward transformation from real-space to reciprocal space.'''
        return Jdag(self, inp)


def read_xyz(filename):
    '''Load atoms objects from xyz files.'''
    # XYZ file definitions: https://en.wikipedia.org/wiki/XYZ_file_format
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    # The first line contains the number of atoms
    Natoms = int(lines[0].strip())
    # The second line can contain a comment, print it if available
    comment = lines[1].strip()
    if comment:
        print(f'XYZ file comment: \'{comment}\'')
    atom = []
    X = []
    # Following lines contain atom positions with the format: Atom x-pos y-pos z-pos
    for line in lines[2:2 + Natoms]:
        split = line.strip().split()
        atom.append(split[0])
        X.append(np.float_(split[1:4]))
    # xyz files are in angstrom, so convert to bohr
    X = ang2bohr(np.asarray(X))
    return atom, X


def write_cube(atoms, field, filename):
    '''Generate cube files from a given (real-space) field.'''
    # It seems, that there is no standard for cube files. The following definition will work with
    # VESTA and is taken from: https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html
    # Atomic units are assumed, so there is no need for conversion.
    atom = atoms.atom
    a = atoms.a
    r = atoms.r
    S = atoms.S
    X = atoms.X
    Z = atoms.Z

    # Our field data has been created in a different order than needed for cube files
    # (triple loop over z,y,x instead of x,y,z), so rearrange it with some index magic.
    idx = []
    for Nx in range(S[0]):
        for Ny in range(S[1]):
            for Nz in range(S[2]):
                idx.append(Nx + Ny * S[0] + Nz * S[0] * S[1])
    idx = np.asarray(idx)

    # Make sure we have real valued data in the correct order
    field = np.real(field[idx])

    with open(filename, 'w') as fp:
        # The first two lines have to be a comment.
        # Print file creation time and informations about the file and program.
        fp.write(f'{ctime()}\n')
        fp.write(f'Cube file generated with PlaineDFT {__version__}\n')
        # Number of atoms (int), and origin of the coordinate system (float)
        # The origin is normally at 0,0,0 but we could move our box, so take the minimum
        fp.write(f'{len(X)}  {min(r[:, 0]):.5f}  {min(r[:, 1]):.5f}  {min(r[:, 2]):.5f}\n')
        # Number of points per axis (int), and vector defining the axis (float)
        # We only have a cubic box, so each vector only has one non-zero component
        fp.write(f'{S[0]}  {a / S[0]:.5f}  0.0  0.0\n')
        fp.write(f'{S[1]}  0.0  {a / S[1]:.5f}  0.0\n')
        fp.write(f'{S[2]}  0.0  0.0  {a / S[2]:.5f}\n')
        # Atomic number (int), atomic charge (float), and atom position (floats) for every atom
        for ia in range(len(X)):
            fp.write(f'{symbol2number[atom[ia]]}  {Z[ia]:.5f}')
            fp.write(f'  {X[ia][0]:.5f}  {X[ia][1]:.5f}  {X[ia][2]:.5f}\n')
        # Field data (float) with scientific formatting
        # We have S[0]*S[1] chunks values with a length of S[2]
        for i in range(S[0] * S[1]):
            # Print every round of values, so we can add empty lines between them
            data_str = '%+1.5e  ' * S[2] % tuple(field[i * S[2]:(i + 1) * S[2]])
            # Print a maximum of 6 values per row
            # Max width for this formatting is 90, since 6*len('+1.00000e-000  ')=90
            fp.write(f'{fill(data_str, width=90)}\n\n')
    return


def save_atoms(atoms, filename, clear=False):
    '''Save atoms objects into a pickle files.'''
    # Remove results  from SCF calculations to save some space
    if clear:
        atoms.W = None
        atoms.psi = None
        atoms.estate = None
        atoms.n = None
        atoms.eewald = None
        atoms.etot = None
    with open(filename, 'wb') as fp:
        dump(atoms, fp, HIGHEST_PROTOCOL)
    return


def load_atoms(filename):
    '''Load atoms objects from pickle files.'''
    with open(filename, 'rb') as fh:
        return load(fh)
