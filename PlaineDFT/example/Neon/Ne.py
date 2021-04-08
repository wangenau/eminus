import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from plainedft import Atoms
from plainedft import SCF
from plainedft.plot import *
from plainedft.tools import *

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
SCF(atoms, n_sd=50, n_cg=50, cgform=1)
