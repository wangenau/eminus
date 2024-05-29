#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from eminus import Atoms, read, SCF
from eminus.dft import get_psi
from eminus.extras import get_fods, remove_core_fods
from eminus.localizer import get_FLO

# # Start by with a DFT calculation for methane
atom, pos = read('CH4.xyz')
atoms = Atoms(atom, pos, ecut=10, center=True)
scf = SCF(atoms)
scf.run()

# # Calculate all FODs
fods_all = get_fods(atoms)
print(f'\nAll FODs:\n{fods_all}')

# # Remove core FODs, since the calculation uses a GTH pseudopotential
fods = remove_core_fods(atoms, fods_all)
print(f'\nCore FODs:\n{fods}')

# # The quality from the FOD guess can vary, but you can use these as a decent guess
# import numpy as np
# fods = [np.array([[10.71617803, 10.75510917, 10.73689087],
#                   [10.82635834,  9.25127336,  9.25068483],
#                   [ 9.24857483, 10.79169744,  9.24052496],
#                   [ 9.25441172,  9.25005662, 10.82402898]]),
#         np.array([])]

# # Write the FODs to an XYZ file to view them
atoms.write('CH4_fods.xyz', fods)

# # Generate the Kohn-Sham orbitals
psi = get_psi(scf, scf.W)

# # Calculate the FLOs
FLO = get_FLO(atoms, psi, fods)

# # Write all FLOs to CUBE files
print('\nWrite cube files:')
for i in range(atoms.occ.Nstate):
    print(f'{i + 1} of {atoms.occ.Nstate}')
    atoms.write(f'CH4_FLO_{i + 1}.cube', FLO[0][0, :, i])

# # All of the functionality above can be achieved with the following workflow function
# from eminus.orbitals import FLO
# FLO = FLO(scf, write_cubes=True)
