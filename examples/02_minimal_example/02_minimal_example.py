#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from eminus import Atoms, SCF

# # Create an `Atoms` object with helium at position (0,0,0)
atoms = Atoms('He', [0, 0, 0])

# # Create a `SCF` object...
scf = SCF(atoms)

# # ...and start the calculation
scf.run()
