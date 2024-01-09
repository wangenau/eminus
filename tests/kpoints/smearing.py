import numpy as np
from eminus import Cell, config, SCF
from eminus.dft import get_epsilon, get_epsilon_band
from eminus.extras import plot_bandstructure
# config.verbose = 'debug'

atoms = Cell('Al', 'fcc', ecut=15, a=7.3, kmesh=(1, 1, 1), bands=6, smearing=0.01)
atoms.occ.Nspin = 1
scf = SCF(atoms, etol=1e-5)
scf.run(betat=1)
print(scf.energies)
print(scf.atoms.occ.f)
