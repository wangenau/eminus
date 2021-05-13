#!/usr/bin/env python3
'''
Atoms object that holds every calculation relevant parameters and outputs.
'''
import numpy as np
from numpy.linalg import inv, det
from .operators import O, L, Linv, K, I, J, Idag, Jdag
from .potentials import init_pot
from .gth_loc import init_gth_loc
from .gth_nonloc import init_gth_nonloc
from .read_gth import read_GTH


class Atoms:
    '''Define an atoms object that holds all necessary calculation parameters.'''
    def __init__(self, atom, a, X, Z, Ns, S=None, f=2, ecut=20, verbose=3, pot='GTH', center=False, truncate=True):
        # Necessary inputs
        if isinstance(atom, str):
            atom = [atom]
        self.atom = atom  # Atom type
        self.a = a        # Lattice constant
        self.X = X        # Core positions
        if isinstance(Z, int):
            Z = [Z] * len(X)
        self.Z = Z                # Charge
        self.Ns = Ns              # Number of states
        self.center = center      # Center molecule in box
        self.truncate = truncate  # Bool to turn off G-vector truncation

        # Necessary inputs with presets
        if S is isinstance(S, int):
            S = S * np.array([1, 1, 1])
        if S is None:
            S = 50 * np.array([1, 1, 1])
        if isinstance(f, int):
            f = f * np.ones(self.Ns)
        self.S = S              # Sampling
        self.f = f              # Occupation
        self.ecut = ecut        # Cut-off energy
        self.verbose = verbose  # Toggle debug output
        self.pot = pot.upper()

        # Parameters that will be built out of the inputs
        self.CellVol = None  # Unit cell volume
        self.R = None        # Unit cell
        self.M = None        # Index matrix
        self.N = None        # Index matrix
        self.r = None        # Sample points in unit cell
        self.G = None        # G-vectors
        self.G2 = None       # Squared magnitudes of G-vectors
        self.Sf = None       # Structure factor
        self.active = None   # Mask for active G-vectors
        self.Gc = None       # Cut G-vectors
        self.G2c = None      # Cut squared magnitudes of G-vectors
        self.setup()

        # FIXME: Add comments to non-local part
        # Parameters used for pseudopotentials
        self.GTH = {}         # Contains GTH parameters in a dictionary
        self.Vloc = None      # Local pseudopotential contribution (dual)
        self.NbetaNL = 0      #
        self.prj2beta = None  #
        self.betaNL = None    #
        self.get_pot()

        # Parameters after SCF calculations
        self.W = None       # Basis functions
        self.psi = None     # States
        self.estate = None  # Energy per state
        self.n = None       # Electronic density
        self.etot = None    # Total energy

    def setup(self):
        # Center molecule by its center of mass in the unit cell
        if self.center:
            X = np.asarray(X)
            com = center_of_mass(X)
            self.X = X - (com - a / 2)

        # Build a cubic unit cell
        if self.a is not None:
            R = self.a * np.eye(3)
            self.R = R
            self.CellVol = np.abs(det(R))  # We only have cubic unit cells for now

        ms = np.arange(0, np.prod(self.S))
        m1 = ms % self.S[0]
        m2 = np.floor(ms / self.S[0]) % self.S[1]
        m3 = np.floor(ms / (self.S[0] * self.S[1])) % self.S[2]
        M = np.array([m1, m2, m3]).T
        self.M = M

        n1 = m1 - (m1 > self.S[0] / 2) * self.S[0]
        n2 = m2 - (m2 > self.S[1] / 2) * self.S[1]
        n3 = m3 - (m3 > self.S[2] / 2) * self.S[2]
        N = np.array([n1, n2, n3]).T
        self.N = N

        r = M @ inv(np.diag(self.S)) @ R.T
        self.r = r

        G = 2 * np.pi * N @ inv(R)
        self.G = G

        G2 = np.sum(G**2, axis=1)
        self.G2 = G2

        Sf = np.sum(np.exp(-1j * G @ self.X.conj().T), axis=1)
        self.Sf = Sf

        # FIXME: Remove old G-vector restriction
        # if any((self.S % 2) != 0) and self.verbose > 0:
        #     print('Odd dimension in S, this is could be bad!')
        # eS = self.S / 2 + 0.5
        # edges = np.nonzero(np.any(np.abs(M - np.ones((np.size(M, axis=0), 1)) @ [eS]) < 1, axis=1))
        # G2mx = np.min(G2[edges])
        # active = np.nonzero(G2 < G2mx / 4)
        if self.truncate:
            active = np.nonzero(G2 <= 2 * self.ecut)
        else:
            active = np.nonzero(G2 >= 0)  # Trivial condition to produce the right shape
        self.active = active

        Gc = G[active]
        G2c = G2[active]

        # idx = np.argsort(G2c, kind='mergesort')
        # Gc = Gc[idx]
        # G2c = G2c[idx]

        self.Gc = Gc
        self.G2c = G2c

    def get_pot(self):
        '''Generate the potentials.'''
        if self.pot == 'GTH':
            for ia in range(len(self.atom)):
                self.GTH[self.atom[ia]] = read_GTH('%s-q%s.gth' % (self.atom[ia], self.Z[ia]))
            self.Vloc = init_gth_loc(self)
            self.NbetaNL, self.prj2beta, self.betaNL = init_gth_nonloc(self)
        elif self.pot in ('HARMONIC', 'COULOMB', 'GE'):
            self.Vloc = init_pot(self)
        else:
            print('ERROR: No potential found for %s' % self.pot)

    def O(self, a):
        '''Overlap operator.'''
        return O(self, a)

    def L(self, a):
        '''Laplacian operator.'''
        return L(self, a)

    def Linv(self, a):
        '''Inverse Laplacian operator.'''
        return Linv(self, a)

    def K(self, a):
        '''Precondition by applying 1/(1+G2) to the input.'''
        return K(self, a)

    def I(self, a):
        '''Forward transformation from real-space to reciprocal space.'''
        return I(self, a)

    def J(self, a):
        '''Backwards transformation from reciprocal space to real-space.'''
        return J(self, a)

    def Idag(self, a):
        '''Conjugated forward transformation from real-space to reciprocal space.'''
        return Idag(self, a)

    def Jdag(self, a):
        '''Conjugated backwards transformation from reciprocal space to real-space.'''
        return Jdag(self, a)


def center_of_mass(coords, weights=None):
    '''Calculate the center of mass for a list of points and their weights.'''
    if not isinstance(weights, (list, np.ndarray)):
        weights = [1] * len(coords)
    com = np.full(len(coords[0]), 0, dtype=float)
    for i in range(len(coords)):
        com += coords[i] * weights[i]
    return com / sum(weights)
