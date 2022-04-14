from eminus import Atoms, SCF
from eminus.addons import KSO, FLO
from eminus.energies import *
from eminus.localizer import *
from eminus.scf import get_n_single
from eminus.tools import *
from eminus.units import *

# Start by with a calculation for neon
atoms = Atoms('Ne', [0, 0, 0])
SCF(atoms)

# Generate Kohn-Sham and Fermi-Löwdin orbitals
KSOs = KSO(atoms)
FLOs = FLO(atoms)

# Generate the single-electron densities
# The orbitals have to be in reciprocal space, so transform them
n_kso = get_n_single(atoms, atoms.J(KSOs, False))
n_flo = get_n_single(atoms, atoms.J(FLOs, False))

# Calculate the self-interaction energies
esic_kso = get_Esic(atoms, n_kso)
print(f'\nKSO-SIC energy = {esic_kso} Hartree')

# The SIC energy will also be saved in the Atoms object
get_Esic(atoms, n_flo)
print(f'FLO-SIC energy = {atoms.energies.Esic} Hartree')
print(f'\nAll energies:\n{atoms.energies}')
