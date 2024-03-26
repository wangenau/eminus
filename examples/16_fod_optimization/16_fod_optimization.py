#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.optimize import minimize

from eminus import Atoms, read, SCF, write
from eminus.energies import get_Esic
from eminus.extras import get_fods, remove_core_fods
from eminus.orbitals import FLO

# # Do a simple calculation for methane
# # Use a very small cutoff energy for a small grid
# # The optimization function will be called hundreds of times, so we are interested in speed for this simple example
atom, pos = read('CH4.xyz')
atoms = Atoms(atom, pos, ecut=1, center=True)
scf = SCF(atoms)
scf.run()

# # Generate an initial guess for the FODs and remove the core
fods = get_fods(atoms, basis='pc-0')
fods = remove_core_fods(atoms, fods)
print(f'\nInitial FODs:\n{fods}')


# # Example implementation for a FOD optimization
# # This implementation works for restricted and unrestricted calculations
def optimize_fods(scf, fods):
    def x2fods(x):
        """Transform a 1d list back to FODs."""
        nfods = [len(fod) for fod in fods]
        fod_up = np.reshape(x[: nfods[0] * 3], (nfods[0], 3))
        if len(nfods) > 1:
            fod_dn = np.reshape(x[nfods[0] * 3 :], (nfods[1], 3))
            return [fod_up, fod_dn]
        return [fod_up]

    def get_sic_energy(x):
        """Wrapper function to calculate the SIC energy from a 1d list of FODs."""
        fods = x2fods(x)
        flo = FLO(scf, fods=fods)
        return get_Esic(scf, scf.atoms.J(flo, full=False))

    # Convert FODs to a list such that SciPy's minimize function can work with them
    x = np.array(fods).flatten()
    # Call the optimizer
    print('\nStart FOD optimization...')
    result = minimize(get_sic_energy, x0=x, method='nelder-mead', tol=1e-4, options={'disp': True})
    # Print the SIC energies
    print(f'\nInitial SIC energy:   {get_sic_energy(x):.6f} Eh')
    print(f'Optimized SIC energy: {get_sic_energy(result.x):.6f} Eh')
    # Return the FODs in a usable format
    return x2fods(result.x)


# # Optimize the FODs
# # You may have to run this a few times since the optimizer can sometimes run into an unphysical solution
fods = optimize_fods(scf, fods)
print(f'\nOptimized FODs:\n{fods}')

# # Write the optimized FODs to a file
write(atoms, 'CH4_fods.xyz', fods=fods)
