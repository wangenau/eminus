# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the Cell generation."""

import numpy as np
from numpy.testing import assert_equal

from eminus import Cell
from eminus.data import LATTICE_VECTORS


def test_ecut_verbose():
    """Test the pass-through arguments."""
    ecut = 15
    verbose = "error"
    cell = Cell("He", "sc", ecut, 20, verbose=verbose)
    assert cell.ecut == ecut
    assert cell.verbose == verbose.upper()


def test_atom():
    """Test the atom setting."""
    cell = Cell("Si", "fcc", 30, 20)
    assert cell.Natoms == 1
    cell = Cell("Si", "diamond", 30, 20)
    assert cell.Natoms == 2
    cell = Cell("Si2", "diamond", 30, 20)
    assert cell.Natoms == 2
    cell = Cell(["Si", "Si"], "diamond", 30, 20)
    assert cell.Natoms == 2


def test_lattice():
    """Test the lattice setting."""
    lattice = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    cell = Cell("Si", lattice, 30, None)
    assert_equal(lattice, cell.a)
    cell = Cell("Si", lattice, 30, 0)
    assert_equal(cell.a, 0)
    cell = Cell("Si", "diamond", 30, None)
    assert_equal(cell.a, LATTICE_VECTORS["fcc"])
    cell = Cell("Si", "diamond", 30, -2)
    assert_equal(cell.a, np.eye(3) - 1)


def test_a():
    """Test the a setting."""
    cell = Cell("Si", "fcc", 30, None)
    assert_equal(cell.a, LATTICE_VECTORS["fcc"])
    assert_equal(cell.pos, 0)
    cell = Cell("Si", "diamond", 30, 0)
    assert_equal(cell.a, 0)
    assert_equal(cell.pos, 0)


def test_basis():
    """Test the basis setting."""
    basis = [[0.0, 0.0, 0.0], [1 / 2, 1 / 3, 1 / 4]]
    cell = Cell("Si", "diamond", 30, 20, basis=basis)
    assert len(cell.pos) == len(basis)
    assert len(basis) == cell.Natoms


def test_bands():
    """Test the bands setting."""
    cell = Cell("Si", "diamond", 30, 0, bands=None)
    assert cell.occ.bands == 4
    cell = Cell("Si", "diamond", 30, 0, bands=5)
    assert cell.occ.bands == 5


def test_kmesh():
    """Test the kmesh setting."""
    cell = Cell("Si", "diamond", 30, 0, kmesh=1)
    assert_equal(cell.kpts.kmesh, 1)
    cell = Cell("Si", "diamond", 30, 1, kmesh=2)
    assert cell.kpts.build().Nk == 8
    cell = Cell("Si", "diamond", 30, 1, kmesh=(1, 3, 2))
    assert cell.kpts.build().Nk == 6


def test_kwargs():
    """Test the kwargs pass-through setting."""
    cell = Cell("Si", "diamond", 30, 0, unrestricted=True, spin=1, charge=1)
    assert cell.unrestricted
    assert cell.spin == 1
    assert cell.charge == 1


def test_s():
    """Test that the sampling respects the symmetry of the cell."""
    a = [3, 3, 1]
    cell = Cell("Be", "hexagonal", 30, a)
    assert_equal(a, cell.s / 5)


def test_Z():
    """Test that the number of bands gets updated when updating Z."""
    cell = Cell("Ar", "sc", 10, 10)
    assert cell.occ.bands == 4
    cell.Z = {"Ar": 18}
    cell.build()
    assert cell.occ.bands == 9


if __name__ == "__main__":
    import inspect
    import pathlib

    import pytest

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
