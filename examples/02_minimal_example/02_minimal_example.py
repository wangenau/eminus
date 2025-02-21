# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from eminus import Atoms, SCF

# # Create an `Atoms` object with helium at position (0,0,0)
atoms = Atoms("He", [0, 0, 0])

# # Create a `SCF` object...
scf = SCF(atoms)

# # ...and start the calculation
scf.run()
