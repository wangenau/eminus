from eminus import Atoms, SCF

# Create an Atoms object with helium at position (0,0,0)
atoms = Atoms('He', [0, 0, 0])

# Start the DFT calculation
SCF(atoms)
