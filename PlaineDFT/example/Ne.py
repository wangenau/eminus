import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from plainedft import Atoms
from plainedft import SCF
from plainedft.scf import getE
from plainedft.plot import *
from plainedft.tools import *
from plainedft.gth_nonloc import calc_Enl, calc_Vnl

#def main():
atom = 'Ne'
lattice = 16
X = np.array([[8, 8, 8]])
Z = 8
Ns = 4
S = 75 * np.array([1, 1, 1])
f = 2
ecut = 25
verbose = 5
pot = 'gth'

atoms = Atoms(atom=atom, a=lattice, X=X, Z=Z, Ns=Ns, S=S, f=f, ecut=ecut, verbose=verbose, pot=pot)
#print(atoms.G)
#SCF(atoms, n_sd=50, n_cg=50, cgform=1)
W = np.random.rand(len(atoms.active[0]), atoms.Ns)# + 1j * randn(len(a.active[0]), a.Ns)
print(calc_Enl(atoms, W))
