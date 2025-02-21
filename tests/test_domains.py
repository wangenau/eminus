# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test domain generation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms
from eminus.domains import domain_cuboid, domain_isovalue, domain_sphere, truncate
from eminus.tools import center_of_mass

atoms = Atoms(
    "CH4",
    (
        (0, 0, 0),
        (1.186, 1.186, 1.186),
        (1.186, -1.186, -1.186),
        (-1.186, 1.186, -1.186),
        (-1.186, -1.186, 1.186),
    ),
).build()


@pytest.mark.parametrize("length", [0.001, 0.01, 0.1, 1, 10])
def test_domain_cuboid(length):
    """Test cuboidal domain generation by ensuring that the grid points are close to the center."""
    out = truncate(atoms.r, domain_cuboid(atoms, length))
    ref = np.zeros_like(out)
    assert_allclose(out, ref, atol=length)
    # Test multiple centers, but use the same center for the test
    com = center_of_mass(atoms.pos)
    out = truncate(atoms.r, domain_cuboid(atoms, length, (com, com)))
    ref = np.zeros_like(out)
    assert_allclose(out, ref, atol=length)


@pytest.mark.parametrize("radius", [0.001, 0.01, 0.1, 1, 10])
def test_domain_sphere(radius):
    """Test spherical domain generation by ensuring that the grid points are close to the center."""
    out = truncate(atoms.r, domain_sphere(atoms, radius))
    ref = np.zeros_like(out)
    assert_allclose(out, ref, atol=radius)
    # Test multiple centers, but use the same center for the test
    com = center_of_mass(atoms.pos)
    out = truncate(atoms.r, domain_sphere(atoms, radius, (com, com)))
    ref = np.zeros_like(out)
    assert_allclose(out, ref, atol=radius)


def test_domain_isovalue():
    """Test isovalue domain execution."""
    out = truncate(atoms.G2, domain_isovalue(atoms.G2, 0.1))
    assert not (out == atoms.G2).all()


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
