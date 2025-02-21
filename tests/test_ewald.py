# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test Ewald summation.

Calculate Madelung constants from Ewald energies for different crystal systems.
Reference: J. Chem. Phys. 28, 164.
"""

import numpy as np
from numpy.testing import assert_allclose

from eminus import Cell
from eminus.energies import get_Eewald


def test_NaCl():
    """Calculate Madelung constant for NaCl."""
    cell = Cell("NaCl", "fcc", ecut=1, a=1, basis=[[0.0, 0.0, 0.0], [1 / 2, 1 / 2, 1 / 2]])
    cell.Z = [1, -1]
    madelung = -0.5 * get_Eewald(cell, gcut=15)
    madelung_ref = 1.7475645946331822
    assert_allclose(madelung, madelung_ref)


def test_CsCl():
    """Calculate Madelung constant for CsCl."""
    cell = Cell("CsCl", "sc", ecut=1, a=1, basis=[[0.0, 0.0, 0.0], [1 / 2, 1 / 2, 1 / 2]])
    cell.Z = [1, -1]
    madelung = -0.5 * np.sqrt(3) * get_Eewald(cell, gcut=15)
    madelung_ref = 1.76267477307099
    assert_allclose(madelung, madelung_ref)


if __name__ == "__main__":
    import inspect
    import pathlib

    import pytest

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
