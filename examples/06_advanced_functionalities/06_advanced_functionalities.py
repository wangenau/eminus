import eminus
from eminus import Atoms, SCF, write_cube
from eminus.dft import get_psi
from eminus.localizer import wannier_cost
from eminus.tools import center_of_mass, check_orthonorm, get_dipole, get_IP
from eminus.units import ebohr2d, ha2kcalmol
import numpy as np

# Start by with a simple DFT calculation for neon
atoms = Atoms('Ne', [0, 0, 0], center=True)
scf = SCF(atoms)
scf.run()

# Calculate the dipole moment
# Make sure that the unit cell is big enough, and that the density does not extend over the borders
# Centering the system is recommended to achieve this
dip = get_dipole(scf)
print(f'\nDipole moments = {dip} a0')
print(f'Total dipole moment = {ebohr2d(np.linalg.norm(dip))} D')

# Calculate ionization potentials
ip = get_IP(scf)
print(f'\nIonization potential = {ha2kcalmol(ip)} kcal/mol\n')

# Transform the orbitals to real-space to get the Kohn-Sham orbitals
# Make sure to use orthogonal wave functions to generate them
psi = atoms.I(get_psi(scf, scf.W))

# Check orthonormality of Kohn-Sham orbitals
print('Orthonormality of Kohn-Sham orbitals:')
check_orthonorm(atoms, psi)

# Some functions are controlled with a global logging level that can be changed with
eminus.log.verbose = 4

# Calculate the orbital variance and spread of the orbitals
cost = wannier_cost(atoms, psi)
print(f'\nOrbital variances = {cost} a0^2')
print(f'Total spread = {np.sum(np.sqrt(cost))} a0')

# Calculate the center of mass of the density
com = center_of_mass(atoms.r, scf.n)
print(f'\nDensity center of mass = {com} a0')
print(f'Neon position = {atoms.X[0]} a0')

# Write all orbitals to cube files
print('\nWrite cube files:')
for i in range(atoms.Ns):
    print(f'{i + 1} of {atoms.Ns}')
    write_cube(atoms, psi[:, i], f'Ne_{i + 1}.cube')
