from eminus import Cell, SCF
from eminus.units import ang2bohr

# # Create a diamond cell
# # `kmesh=(2, 1, 1)` will create a Monkhorst-Pack k-point grid with two k-points
cell = Cell('C', 'diamond', 30, ang2bohr(3.57), kmesh=(2, 1, 1))

# # One can also use a custom crystal basis, the diamond system would be equivalent of the following
# cell = Cell('C', 'fcc', 30, ang2bohr(3.57), basis=[[0, 0, 0], [0.25, 0.25, 0.25]])

# # Do a DFT calculation for the cell
scf = SCF(cell)
scf.run()

# # The `SCF` and `Atoms` objects have a k-point object
print(f'\nKPoints object\n{scf.kpts}\n')

# # The object can be modified, e.g., to create a non-Gamma-point centered mesh or to shift the k-points (do not forget to rebuild the object)
print(f'Original points:\n{cell.kpts.k}')
cell.kpts.gamma_centered = False
cell.build()
print(f'non-Gamma-centered points:\n{cell.kpts.k}')
cell.kpts.kshift = [1, 0, 0]
cell.kpts.build()
print(f'Shifted points:\n{cell.kpts.k}\n')

# # One can also use custom k-points
# # The easiest way to set them is the following
cell.set_k([0, 0, 0])
SCF(cell, verbose=3).run()
