from eminus import Atoms, SCF

# # Create an `Atoms` object with helium at position (0,0,0)
atoms = Atoms('He', [0, 0, 0], Nspin=1)

# # Create a `SCF` object...
scf = SCF(atoms)

# # ...and start the calculation
scf.run()
