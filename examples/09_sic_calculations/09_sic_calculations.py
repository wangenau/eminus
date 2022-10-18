from eminus import Atoms, SCF
from eminus.energies import get_Esic
from eminus.orbitals import FLO, KSO

# # Start with a DFT calculation for neon
atoms = Atoms('Ne', [0, 0, 0], ecut=10, Nspin=1).build()
scf = SCF(atoms)
scf.run()

# # Generate Kohn-Sham and Fermi-Löwdin orbitals
KSO = KSO(scf)
FLO = FLO(scf)

# # Calculate the self-interaction energies
# # The orbitals have to be in reciprocal space, so transform them
esic_kso = get_Esic(scf, atoms.J(KSO, False))
print(f'\nKSO-SIC energy = {esic_kso} Eh')

# # The SIC energy will also be saved in the `SCF` object
# # The quality of the FLO-SIC energy will vary with the FOD guess
get_Esic(scf, atoms.J(FLO, False))
print(f'FLO-SIC energy = {scf.energies.Esic} Eh')
print(f'\nAll energies:\n{scf.energies}')
