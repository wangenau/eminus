from eminus import Cell, RSCF
from eminus.tools import get_Efermi
from eminus.units import ha2kelvin

# # Create a cell for an aluminium crystal
cell = Cell('Al', 'fcc', ecut=10, a=7.63075, bands=6, smearing=0.01)

# # Create the SCF object
# # The occupations update per SCF cycle can be controlled with the `smear_update` parameter
scf = RSCF(cell, etol=1e-5, verbose=0)
scf.smear_update = 1

# # Do the DFT calculation
scf.run(betat=1e-3)

# # The resulting total energy includes an entropic term when smearing is enabled
print(scf.energies)

# # We can extrapolate the energy to T=0
print(f'\nEtot({ha2kelvin(cell.occ.smearing):.2f} K) = {scf.energies.Etot:.6f} Eh')
print(f'Etot(0 K)       = {scf.energies.extrapolate():.6f} Eh')

# # Calculate the Fermi energy
print(f'\nEf = {get_Efermi(scf):.6f} Eh')
