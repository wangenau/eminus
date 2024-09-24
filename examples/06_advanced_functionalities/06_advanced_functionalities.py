# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
import numpy as np

import eminus
from eminus import Atoms, SCF
from eminus.dft import get_psi
from eminus.localizer import wannier_cost
from eminus.tools import center_of_mass, check_orthonorm, get_dipole, get_ip
from eminus.units import ebohr2d, ha2kcalmol

# # Start with a simple DFT calculation for neon
# # If one needs cell parameters from the `Atoms` object one can use the `atoms.build` function to generate them
# # eminus also supports GGA functionals like the Chachiyo GGA
# # A different cg-form can be used if the system is hard to converge
atoms = Atoms("Ne", [0, 0, 0], ecut=10, center=True)
scf = SCF(atoms, xc="chachiyo")
scf.run(cgform=3)

# # Calculate the dipole moment
# # Make sure that the cell is big enough, and that the density does not extend over the borders
# # Centering the system is recommended to achieve this
dip = get_dipole(scf)
print(f"\nDipole moments = {dip} a0")
print(f"Total dipole moment = {ebohr2d(np.linalg.norm(dip))} D")

# # Calculate ionization potentials
ip = get_ip(scf)
print(f"\nIonization potential = {ha2kcalmol(ip)} kcal/mol\n")

# # Transform the orbitals to real-space to get the Kohn-Sham orbitals
# # Make sure to use orthogonal wave functions to generate them
psi = atoms.I(get_psi(scf, scf.W))

# # Some functions are controlled with a global logging level that can be changed with
eminus.config.verbose = 3

# # Check orthonormality of Kohn-Sham orbitals
print("Orthonormality of Kohn-Sham orbitals:")
check_orthonorm(atoms, psi)

# # Calculate the orbital variance and spread of the orbitals
cost = wannier_cost(atoms, psi)
print(f"\nOrbital variances = {cost} a0^2")
print(f"Total spread = {np.sum(np.sqrt(cost))} a0")

# # Calculate the center of mass of the density
com = center_of_mass(atoms.r, scf.n)
print(f"\nDensity center of mass = {com} a0")
print(f"Neon position = {atoms.pos[0]} a0")

# # Write all orbitals to CUBE files
print("\nWrite cube files:")
for i in range(atoms.occ.Nstate):
    print(f"{i + 1} of {atoms.occ.Nstate}")
    atoms.write(f"Ne_{i + 1}.cube", psi[0][0, :, i])

# # Another useful setting is the number of threads
print(f"\nThreads: {eminus.config.threads}\n")

# # You can also set them and check the configuration afterwards
eminus.config.threads = 2
eminus.config.info()

# # eminus uses the NumPy and SciPy packages
# # These packages will listen to the OMP_NUM_THREADS and/or MKL_NUM_THREADS flags
# # Setting these flags will control how many threads eminus uses.
