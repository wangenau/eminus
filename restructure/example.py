import numpy as np
from atoms import Atoms
from scf import SCF
from plot import plot_pot, plot_n
from potential import Arias_pot
from gth import GTH_pot

# atom = 'Ge'
# lattice = 5.66 / 0.52917721
# X = np.array([[0, 0, 0]])
# Z = 4
# Ns = 4
# S = np.array([48, 48, 48])
# f = np.array([2, 2 / 3, 2 / 3, 2 / 3])
# ecut = 800
# verbose = 3

# atom = 'He'
# lattice = 16
# X = np.array([[0, 0, 0]])
# Z = 2
# Ns = 2
# S = np.array([64, 64, 64])
# f = 2
# ecut = 800
# verbose = 3

atom = 'H'
lattice = 16
# X = np.array([[0, 0, 0], [1.5, 0, 0]])
X = np.array([[8, 8, 8]])
Z = 1
Ns = 1
S = np.array([64, 64, 64])
f = 1
from constants import HARTREE
ecut = 50 * HARTREE / 2
ecut = 400
#print(ecut)
verbose = 3

a = Atoms(atom=atom, a=lattice, X=X, Z=Z, Ns=Ns, S=S, f=f, ecut=ecut, verbose=verbose)
Vps, Vdual = Arias_pot(a)
#Vps, Vdual = GTH_pot(a)

a.Vdual = Vdual
#plot_pot(a)

SCF(a, Nit_sd=0, Nit=50, cgform=1)
# print('4s-4p energy: %f (NIa.ST: %f)' % (a.epsilon[0] - a.epsilon[1], -0.276641))

#plot_n(a)
