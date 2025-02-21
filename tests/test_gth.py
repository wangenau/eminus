# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test GTH functions."""

import inspect
import numbers
import pathlib

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from eminus import Atoms, Cell, SCF


def test_GTH():
    """Test the GTH object."""
    cell = Cell("Ne", "sc", 20, 10, kmesh=(1, 1, 2))
    scf = SCF(cell)
    gth = scf.gth
    gth["Ne"]  # Test that the object can be accessed using square brackets
    print(gth)  # Test that the object can be printed
    assert gth.NbetaNL == 5
    assert len(gth.betaNL) == cell.kpts.Nk
    assert gth.betaNL[0].shape == (4337, 5)


def test_norm():
    """Test the norm of the GTH projector functions."""
    # Actinium is the perfect test case since it tests all cases in eval_proj_G with the minimum
    # number of electrons (q-11)
    # The norm depends on the cut-off energy but will converge to 1
    # Choose an ecut such that we get at least 0.9
    atoms = Atoms("Ac", (0, 0, 0), ecut=35)
    scf = SCF(atoms)
    for ik in range(scf.kpts.Nk):
        for i in range(scf.gth.NbetaNL):
            norm = np.sum(scf.gth.betaNL[ik][:, i] * scf.gth.betaNL[ik][:, i])
            assert_allclose(abs(norm), 1, atol=1e-1)


def test_mock():
    """Test that ghost atoms have no contribution."""
    # Reference calculation without ghost atom
    atoms = Atoms("He", (0, 0, 0), ecut=1)
    E_ref = SCF(atoms).run()
    # Calculation with ghost atom
    atoms = Atoms("HeX", [(0, 0, 0), (1, 0, 0)], ecut=1)
    scf = SCF(atoms)
    E = scf.run()
    assert_allclose(E, E_ref)
    for key in scf.gth["X"]:
        assert_equal(scf.gth["X"][key], 0)


def test_custom_files():
    """Test the option to use custom GTH files in Atoms and SCF."""
    file_path = str(pathlib.Path(inspect.stack()[0][1]).parent)
    atoms = Atoms("B", (0, 0, 0)).build()
    atoms.Z = file_path
    assert isinstance(atoms.Z[0], numbers.Real)
    assert atoms.Z[0] == 3
    scf = SCF(atoms, pot=file_path)
    assert scf.gth["B"]["rloc"] == 0.41878773


def test_multiple_files():
    """Test atoms with multiple GTH files."""
    cell = Cell("Ga", "sc", 10, 10)
    assert cell.Z[0] == 3


if __name__ == "__main__":
    import pytest

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
