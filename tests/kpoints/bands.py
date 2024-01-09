from eminus import Cell, config, SCF
from eminus.extras import plot_bandstructure
# config.verbose = 'debug'

atoms = Cell('Si', 'diamond', ecut=15, a=10.2631, kmesh=(1, 1, 1), bands=8, smearing=0)
# atoms.occ.Nspin = 2
scf = SCF(atoms, etol=1e-5)
scf.run()
scf.kpts.path = 'LGXU,KG'
scf.kpts.Nk = 25

scf.converge_bands()
plot_bandstructure(scf)
