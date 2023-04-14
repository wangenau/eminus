#!/usr/bin/env python3
'''Test different energy contributions.'''
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF

# The reference contributions are similar for the polarized and unpolarized case,
# but not necessary the same (for bad numerics)
E_ref = {
    'Ekin': 12.034539521,
    'Ecoul': 17.843659217,
    'Exc': -4.242215291,
    'Eloc': -58.537341024,
    'Enonloc': 8.152487157,
    'Eewald': -4.539675967,
    'Esic': -0.392074710,
    'Etot': -29.680621097
}

# Run the spin-unpaired calculation at first
atoms_unpol = Atoms('Ne', (0, 0, 0), s=20, ecut=10, Nspin=1)
scf_unpol = SCF(atoms_unpol, sic=True, min={'pccg': 40})
scf_unpol.run()
# Do the spin-paired calculation afterwards
# Use the orbitals from the restricted calculation as an initial guess for the unrestricted case
# This saves time and ensures we run into the same minimum
atoms_pol = Atoms('Ne', (0, 0, 0), s=20, ecut=10, Nspin=2)
scf_pol = SCF(atoms_pol, sic=True, min={'pccg': 40})
scf_pol.W = np.array([scf_unpol.W[0] / 2, scf_unpol.W[0] / 2])
scf_pol.run()


@pytest.mark.parametrize('energy', E_ref.keys())
def test_minimizer_unpol(energy):
    '''Check the spin-unpaired energy contributions.'''
    E = getattr(scf_unpol.energies, energy)
    assert_allclose(E, E_ref[energy], atol=1e-4)


@pytest.mark.parametrize('energy', E_ref.keys())
def test_minimizer_pol(energy):
    '''Check the spin-paired energy contributions.'''
    E = getattr(scf_pol.energies, energy)
    assert_allclose(E, E_ref[energy], atol=1e-4)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
