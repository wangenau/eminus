import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from plainedft import Atoms
from plainedft import SCF
from plainedft.plot import *
from plainedft.tools import *
from plainedft.atoms_io import *

atom = 'H'
lattice = 16
X = np.array([[8, 8, 8]])
Z = 1
Ns = 1
S = 64 * np.array([1, 1, 1])
f = 1
ecut = ev2hartree(800)
verbose = 4
pot = 'gth'

atoms = Atoms(atom=atom, a=lattice, X=X, Z=Z, Ns=Ns, S=S, f=f, ecut=ecut, verbose=verbose, pot=pot)

SCF(atoms, n_sd=0, n_cg=50, cgform=1)
save_atoms(atoms, 'hydrogen.pkl')

# atoms = load_atoms('hydrogen.pkl')
# print(atoms.estate)
# #plot_den(atoms)
# plot_den_iso(atoms, 50, 20)
