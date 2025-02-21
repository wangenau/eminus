# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from eminus import Atoms, read, SCF
from eminus.units import bohr2ang

# # Some file standards are supported to be read from
atom, pos = read("CH4.xyz")

# # To immediately create an `Atoms` object you can do the following
atoms = Atoms(*read("CH4.xyz"))

# # CUBE files are supported as well
# # Here, lattice information and field information are given as well
atom, pos, Z, a, s, field = read("CH4.cube")

# # Create an `Atoms` object with it and start a DFT calculation
atoms = Atoms(atom=atom, pos=pos, a=a)
atoms.s = s
scf = SCF(atoms)
scf.run()

# # Write the total density to a CUBE file, e.g., to visualize it
scf.write("CH4_density.cube", scf.n)

# # Please note that XYZ files will use Angstrom as length units
# # CUBE files have no standard, but atomic units will be assumed
# # Units can always be converted using the unit conversion functionality
print(f"\nMethane coordinates in Bohr:\n{pos}")
print(f"\nMethane coordinates in Angstrom:\n{bohr2ang(pos)}")

# # You can also save the `Atoms` or `SCF` object directly to load them later
# # `read` and `write` are unified functions that determine the corresponding function using the file ending
atoms.write("CH4")
atoms = read("CH4.json")
