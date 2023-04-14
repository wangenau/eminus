#!/usr/bin/env python3
'''Test different minimization functions.'''
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.minimizer import IMPLEMENTED

E_ref = -1.960545545  # The reference value is the same for the polarized and unpolarized case
tolerance = 1e-9
atoms_unpol = Atoms('He', (0, 0, 0), s=10, ecut=10, Nspin=1)
atoms_pol = Atoms('He', (0, 0, 0), s=10, ecut=10, Nspin=2)
# Restart the calculations from a previous result
# This way we make sure we run into the same minima and we have no convergence problems
scf_unpol = SCF(atoms_unpol, etol=tolerance)
scf_pol = SCF(atoms_pol, etol=tolerance)


@pytest.mark.parametrize('minimizer', IMPLEMENTED.keys())
def test_minimizer_unpol(minimizer):
    '''Check the spin-unpaired minimizer functions.'''
    scf_unpol.min = {minimizer: 100}
    if 'cg' in minimizer:
        for i in range(1, 4):
            scf_unpol.cgform = i
            E = scf_unpol.run()
            assert_allclose(E, E_ref, atol=tolerance)
    else:
        E = scf_unpol.run()
        assert_allclose(E, E_ref, atol=tolerance)


@pytest.mark.parametrize('minimizer', IMPLEMENTED.keys())
def test_minimizer_pol(minimizer):
    '''Check the spin-paired minimizer functions.'''
    scf_pol.min = {minimizer: 100}
    if 'cg' in minimizer:
        for i in range(1, 4):
            scf_pol.cgform = i
            E = scf_pol.run()
            assert_allclose(E, E_ref, atol=tolerance)
    else:
        E = scf_pol.run()
        assert_allclose(E, E_ref, atol=tolerance)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
