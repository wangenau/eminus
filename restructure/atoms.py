#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm, inv
from operators import O, L, Linv, K, I, J, Idag, Jdag
from constants import HARTREE


class Atoms:
    '''Define an atoms object that holds all necessary calculation parameters.'''
    def __init__(self, atom, a, X, Z, Ns, S=None, f=2, ecut=600, verbose=3):
        # Necessary inputs
        self.atom = atom  # Atom type
        self.a = a        # Lattice constant
        self.X = X        # Core positions
        self.Z = Z        # Charge
        self.Ns = Ns      # Number of electrons

        # Necessary inputs with presets
        if S is None:
            S = 50 * np.ones(3)
        if isinstance(f, int):
            f = f * np.ones(self.Ns)
        self.S = S              # Sampling
        self.f = f              # Occupation
        self.ecut = ecut        # Cut-off energy
        self.verbose = verbose  # Toggle debug output

        # Parameters that will be built out of the inputs
        self.R = None       # Unit cell
        self.M = None       # Index matrix
        self.N = None       # Index matrix
        self.r = None       # Sample points in unit cell
        self.G = None       # G-vectors
        self.G2 = None      # Squared magnitudes of G-vectors
        self.Sf = None      # Structure factor
        self.active = None  # Mask for active G-vectors
        self.G2c = None     # Cut squared magnitudes of G-vectors
        self.setup()

        # Parameters after SCF calculations
        self.psi = None      # States
        self.epsilon = None  # Energy per state
        self.n = None        # Electronic density
        self.etot = None     # Total energy

    def setup(self):
        # Build a cubic unit cell
        if self.a is not None:
            R = self.a * np.eye(3)
            self.R = R

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

        if any((self.S % 2) != 0):
            print('Odd dimension in S, this is really bad!')
        eS = self.S / 2 + 0.5
        edges = np.nonzero(np.any(np.abs(M - np.ones((np.size(M, axis=0), 1)) @ [eS]) < 1, axis=1))
        G2mx = np.min(G2[edges])
        # active = np.nonzero(G2 < G2mx / 4)
        active = np.nonzero(G2 < 2 * self.ecut / HARTREE)
        self.active = active

        G2c = G2[active]
        self.G2c = G2c

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
