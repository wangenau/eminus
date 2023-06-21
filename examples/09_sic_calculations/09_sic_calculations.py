from eminus import Atoms, SCF
from eminus.energies import get_Esic
from eminus.orbitals import FLO

# # Start with a DFT calculation for neon including a SIC calculation
atoms = Atoms('Ne', [0, 0, 0], ecut=10)
scf = SCF(atoms, sic=True)
scf.run()

# # Generate Kohn-Sham and Fermi-Loewdin orbitals
FLO = FLO(scf)

# # Print the self-interaction energy from the `SCF` object
print(f'\nKSO-SIC energy = {scf.energies.Esic} Eh')

# # The quality of the FLO-SIC energy will vary with the FOD guess
# # The one-shot FLO-SIC energy should be lower than the KSO-SIC one
esic = get_Esic(scf, atoms.J(FLO, False))
print(f'FLO-SIC energy = {esic} Eh')
# # The SIC energy will also be saved in the `SCF` object
print(f'\nAll energies:\n{scf.energies}')
