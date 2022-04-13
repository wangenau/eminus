import numpy as np

from eminus import Atoms, SCF, write_cube
from eminus.localizer import wannier_cost
from eminus.tools import center_of_mass, get_dipole, get_IP, check_orthonorm
from eminus.units import ebohr2d, ha2kcalmol

# Start by with a simple calculation for neon
atoms = Atoms('Ne', [0, 0, 0], center=True, verbose=2)
SCF(atoms)
atoms.verbose = 1

# Calculate the dipole moment
# Make sure that the unit cell is big enough, and that the density does not extend over the borders
# Centering the system is reccomended to achieve this
dip = get_dipole(atoms)
print(f'\nDipole moments = {dip} e/Bohr')
print(f'Total dipole moment = {ebohr2d(np.linalg.norm(dip))} Debye')

# Calculate ionization potentials
ip = get_IP(atoms)
print(f'\nIonization potential = {ha2kcalmol(ip)} kcal/mol\n')

# Transform the orbitals to real-space to get the Kohn-Sham orbitals
psi = atoms.I(atoms.W)

# Check orthonormality of Kohn-Sham orbitals
print('Orthonormality of Kohn-Sham orbitals:')
check_orthonorm(atoms, psi)

# Calculate the orbital variance and spread of the orbitals
var = wannier_cost(atoms, psi)
print(f'\nOrbital variances = {var} Bohr^2')
print(f'Total spread = {np.sum(np.sqrt(var))} Bohr')

# Calculate the center of mass of the density
com = center_of_mass(atoms.r, atoms.n)
print(f'\nDensity center of mass = {com} Bohr')
print(f'Neon position = {atoms.X[0]} Bohr')

# Write all orbitals to cube files
print('\nWrite cube files:')
for i in range(atoms.Ns):
    print(f'{i + 1} of {atoms.Ns}')
    write_cube(atoms, psi[:, i], f'Ne_{i + 1}.cube')
