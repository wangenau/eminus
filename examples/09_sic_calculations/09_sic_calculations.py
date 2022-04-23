from eminus import Atoms, SCF
from eminus.energies import get_Esic
from eminus.orbitals import FLO, KSO

# Start by with a DFT calculation for neon
atoms = Atoms('Ne', [0, 0, 0])
SCF(atoms)

# Generate Kohn-Sham and Fermi-Löwdin orbitals
KSO = KSO(atoms)
FLO = FLO(atoms)

# Calculate the self-interaction energies
# The orbitals have to be in reciprocal space, so transform them
esic_kso = get_Esic(atoms, atoms.J(KSO, False))
print(f'\nKSO-SIC energy = {esic_kso} Eh')

# The SIC energy will also be saved in the Atoms object
# The quality of the FLO-SIC energy will vary with the FOD guess
get_Esic(atoms, atoms.J(FLO, False))
print(f'FLO-SIC energy = {atoms.energies.Esic} Eh')
print(f'\nAll energies:\n{atoms.energies}')
