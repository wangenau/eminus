from eminus import Cell
from eminus.extras import view

atoms = Cell('Si', 'diamond', ecut=15, a=10.2631, kmesh=2, bands=8, smearing=0).build()
atoms.kpts.path = 'LGXU,KGW'
atoms.kpts.Nk = 50
atoms.kpts.build()
view(atoms)
view(atoms.kpts)
