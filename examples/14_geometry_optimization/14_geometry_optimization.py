from scipy.optimize import minimize_scalar

from eminus import Atoms, SCF
from eminus.units import bohr2ang, ha2ev


# # Create a simple cost function for the optimizer
# # The objective is of course the total energy of the molecule
# # Displace one hydrogen by a distance `d`, choose a small `ecut` for a faster evaluation
def cost(d):
    atoms = Atoms('H2', [[0, 0, 0], [0, 0, d]], ecut=5)
    scf = SCF(atoms, verbose=0)
    etot = scf.run()
    print(f'd={bohr2ang(d):.6f} A    Etot={ha2ev(etot):.6f} eV')
    return etot


# # Since we only optimize one value here, we use the minimize scalar function
# # We also add bounds since d=0 should be forbidden
# # When optimizing multiple coordinates one can give a reasonable start configuration instead of giving explicit bounds
res = minimize_scalar(cost, bounds=(0.1, 10))
print(f'Optimized bond length: {bohr2ang(res.x):.6f} A')

# # The same could be done, e.g., for the lattice constant of a germanium solid as seen in example 12
