import numpy as np

from eminus import Atoms, SCF, write
from eminus.units import ang2bohr, ev2ha

# # Create a germanium crystal with an fcc crystal structure
# # Here, `a` is not only the unit cell size but the lattice constant as well, defining the periodicity
# # As a reminder, the center option should not be used when explicitly using periodicity
# # Since electronvolt and Angstrom are common in solid-state physics convert the units here to atomic units
atom = 'Ge8'
a = ang2bohr(5.658)
ecut = ev2ha(500)
X = a * np.array([
    [0, 0, 0],
    [0, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.5, 0, 0.5],
    [0.5, 0.5, 0],
    [0.75, 0.25, 0.25],
    [0.75, 0.75, 0.25]
])
atoms = Atoms(atom=atom, X=X, a=a, ecut=ecut, Nspin=1)

# # Use the pseudopotential from Tomas Arias (only local and only for germanium)
# # The GTH pseudopotential will work as well and would include non-local effects
# # Reduce the convergence tolerance for this example
scf = SCF(atoms, pot='Ge', etol=1e-4)
scf.run()

# # Save the density as a cube file
write(atoms, 'Ge_solid_density.cube', scf.n)

# # If you have the viewer extra installed, the following lines will plot the unit cell and 20 isosurfaces of the density, such that 50% of the density is contained
# from eminus.extras import view
# view(scf, plot_n=True, surfaces=20, percent=50)
