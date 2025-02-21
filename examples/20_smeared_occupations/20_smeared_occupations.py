# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from eminus import Cell, RSCF
from eminus.tools import get_Efermi
from eminus.units import ha2kelvin

# # Create a cell for an aluminum crystal
# # Set a smearing value in atomic units
# # The resulting SCF will use a Fermi-Dirac function to smear the occupations over the selected bands
cell = Cell("Al", "fcc", ecut=10, a=7.63075, bands=6, smearing=0.01)

# # Create the SCF object
# # The occupations update per SCF cycle can be controlled with the `smear_update` parameter
scf = RSCF(cell, etol=1e-5, verbose=0)
scf.smear_update = 1

# # Do the DFT calculation
scf.run()

# # The resulting total energy includes an entropic term when smearing is enabled
print(scf.energies)

# # We can extrapolate the energy to T=0
print(f"\nEtot({ha2kelvin(cell.occ.smearing):.2f} K) = {scf.energies.Etot:.6f} Eh")
print(f"Etot({0:>7.2f} K) = {scf.energies.extrapolate():.6f} Eh")

# # Calculate the Fermi energy
print(f"\nEf = {get_Efermi(scf):.6f} Eh")
