#!/usr/bin/env python3
"""Test GTH functions."""
import inspect
import pathlib

import numpy as np
from numpy.testing import assert_allclose

from eminus import Atoms, SCF


def test_norm():
    """Test the norm of the GTH projector functions."""
    # Actinium is the perfect test case since it tests all cases in eval_proj_G with the minimum
    # number of electrons (q-11)
    # The norm depends on the cut-off energy but will converge to 1
    # Choose an ecut such that we get at least 0.9
    atoms = Atoms('Ac', (0, 0, 0), ecut=35)
    scf = SCF(atoms)
    for i in range(scf.NbetaNL):
        norm = np.sum(scf.betaNL[:, i] * scf.betaNL[:, i])
        assert_allclose(abs(norm), 1, atol=1e-1)


def test_mock():
    """Test that ghost atoms have no contribution."""
    # Reference calculation without ghost atom
    atoms = Atoms('He', [(0, 0, 0)], s=10)
    E_ref = SCF(atoms).run()
    # Calculation with ghost atom
    atoms = Atoms('HeX', [(0, 0, 0), (1, 0, 0)], s=10)
    scf = SCF(atoms)
    E = scf.run()
    assert_allclose(E, E_ref)
    for key in scf.GTH['X']:
        assert_allclose(scf.GTH['X'][key], 0)


def test_custom_files():
    """Test the option to use custom GTH files in Atoms and SCF."""
    file_path = str(pathlib.Path(inspect.getfile(inspect.currentframe())).parent)
    atoms = Atoms('B', (0, 0, 0), Z=file_path).build()
    assert atoms.Z[0] == 3
    scf = SCF(atoms, pot=file_path)
    assert scf.GTH['B']['rloc'] == 0.42188166


if __name__ == '__main__':
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
