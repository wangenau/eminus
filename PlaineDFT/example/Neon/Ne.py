import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from plainedft import Atoms
from plainedft import SCF
from plainedft.scf import getE
from plainedft.plot import *
from plainedft.tools import *
from plainedft.gth_nonloc import calc_Enl, calc_Vnl

atom = 'Ne'
lattice = 16# + 16/(30-1)
X = np.array([[8, 8, 8]])
Z = 8
Ns = 4
S = 75 * np.array([1, 1, 1])
f = 2
ecut = 25 #ev2hartree(650)
verbose = 5
pot = 'gth'

atoms = Atoms(atom=atom, a=lattice, X=X, Z=Z, Ns=Ns, S=S, f=f, ecut=ecut, verbose=verbose, pot=pot)
#atoms.NbetaNL = 0
SCF(atoms, n_sd=50, n_cg=50, cgform=1)
