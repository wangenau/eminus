import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from plainedft import Atoms
from plainedft import SCF
from plainedft.plot import *
from plainedft.tools import *

# atom = 'Ge'
# lattice = 5.66 / 0.52917721
# X = 0.5 * lattice * np.array([[1, 1, 1]])
# Z = 4
# Ns = 4
# S = np.array([48, 48, 48])
# f = np.array([2, 2 / 3, 2 / 3, 2 / 3])
# ecut = 800
# verbose = 4
# pot = 'ge'

# atom = 'Cl'
# lattice = 16
# X = np.array([[0, 0, 0]])
# Z = 7
# Ns = 4
# S = np.array([48, 48, 48])
# f = np.array([2, 5 / 3, 5 / 3, 5 / 3])
# ecut = 15
# verbose = 5
# pot = 'gth'

atom = 'Ne'
lattice = 16# + 16/(30-1)
X = np.array([[8, 8, 8]])
Z = 8
Ns = 4
S = 20 * np.array([1, 1, 1])
f = 2
ecut = 25 #ev2hartree(650)
verbose = 5
pot = 'gth'

# atom = 'Li'
# lattice = 16
# X = np.array([[0, 0, 0], [0, 0, 5.099]])
# Z = 2
# Ns = 1
# S = np.array([48, 48, 48])
# f = 2
# ecut = ev2hartree(800)
# verbose = 4
# pot = 'gth'

# atom = 'He'
# lattice = 16
# X = np.array([[0, 0, 0]])
# Z = 2
# Ns = 2
# S = np.array([64, 64, 64])
# f = 2
# ecut = 15
# verbose = 4
# pot = 'gth'

# atom = 'H'
# lattice = 16
# X = np.array([[8, 8, 8]])
# Z = 1
# Ns = 1
# S = 64 * np.array([1, 1, 1])
# f = 1
# ecut = ev2hartree(800)
# verbose = 4
# pot = 'gth'

# atom = None
# lattice = 6
# X = np.array([[0, 0, 0]])
# Z = 8
# Ns = 4
# S = np.array([20, 25, 30])
# f = 2
# ecut = ev2hartree(800)
# verbose = 4
# pot='harmonic'

atoms = Atoms(atom=atom, a=lattice, X=X, Z=Z, Ns=Ns, S=S, f=f, ecut=ecut, verbose=verbose, pot=pot)

# plot_pot(atoms)
SCF(atoms, n_sd=0, n_cg=50, cgform=1)
# print(atoms.estate)
# plot_den(atoms)
# plot_den_iso(atoms, 50, 20)
