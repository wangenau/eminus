# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the k-points functionalities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from eminus.data import LATTICE_VECTORS, SPECIAL_POINTS
from eminus.kpoints import (
    bandpath,
    gamma_centered,
    get_brillouin_zone,
    KPoints,
    kpoints2axis,
    monkhorst_pack,
)


def test_lattice():
    """Test the setting of lattice."""
    kpts = KPoints("fcc")
    print(kpts)  # Test that the object can be printed
    assert kpts.lattice == "fcc"


@pytest.mark.parametrize(
    ("a", "ref"), [(None, np.eye(3)), (2, 2 * np.eye(3)), (np.ones((3, 3)), np.ones((3, 3)))]
)
def test_a(a, ref):
    """Test the setting of a."""
    kpts = KPoints("sc", a).build()
    assert_equal(kpts.a, ref)


@pytest.mark.parametrize(("kmesh", "ref"), [(None, None), (2, [2] * 3), ([1, 2, 3], [1, 2, 3])])
def test_kmesh(kmesh, ref):
    """Test the setting of kmesh."""
    kpts = KPoints("fcc")
    kpts.kmesh = kmesh
    kpts.build()
    assert kpts.path is None
    assert_equal(kpts.kmesh, ref)
    assert_allclose(np.sum(kpts.wk), 1)
    assert len(kpts.wk) == kpts.Nk
    assert len(kpts.k) == kpts.Nk


def test_kshift():
    """Test the setting of kshift."""
    kpts = KPoints("fcc")
    kpts.kshift = [1] * 3
    kpts.build()
    assert_equal(kpts.k, 1)


def test_gamma_centered():
    """Test the setting of gamma_centered."""
    kpts = KPoints("fcc")
    kpts.gamma_centered = True
    kpts.kmesh = 2
    kpts.build()
    assert_equal(kpts.k[0], 0)
    kpts.gamma_centered = False
    kpts.build()
    assert np.any(kpts.k[0] != 0)


@pytest.mark.parametrize(("path", "ref"), [("G", 0), ("gX", [[0, 0, 0], [0, 2 * np.pi, 0]])])
def test_path(path, ref):
    """Test the setting of path."""
    kpts = KPoints("fcc")
    kpts.path = path
    kpts.build()
    assert kpts.kmesh is None
    assert_allclose(kpts.k, ref)


def test_k_scaled():
    """Test the setting of k_scaled."""
    kpts = KPoints("sc")
    kpts.kmesh = 2
    kpts.build()
    assert_allclose(np.abs(kpts.k_scaled - 1 / 4), 1 / 4)


def test_monkhorst_pack_generation():
    """Test the Monkhorst-Pack mesh generation."""
    k_points = monkhorst_pack((1, 1, 1))
    assert_equal(k_points, 0)
    k_points = monkhorst_pack((2, 2, 2))
    assert_equal(np.abs(k_points), 1 / 4)


def test_gamma_centered_generation():
    """Test the Gamma centered mesh generation."""
    k_points = gamma_centered((1, 1, 1))
    assert_equal(k_points, 0)
    k_points = gamma_centered((2, 2, 2))
    assert_equal(k_points[0], 0)
    assert np.all(k_points >= 0)


def test_bandpath_lgx():
    """Test a simple band path in the FCC lattice."""
    s_points = [
        SPECIAL_POINTS["fcc"]["L"],
        SPECIAL_POINTS["fcc"]["G"],
        SPECIAL_POINTS["fcc"]["X"],
    ]

    kpts = KPoints("fcc", LATTICE_VECTORS["fcc"])
    kpts.path = "LGX"
    kpts.Nk = 2  # Test that that Nk gets set to 3, since 3 special points are set
    k_points = bandpath(kpts.build())
    assert len(k_points) == 3
    assert_equal(k_points, s_points)

    kpts.Nk = 10
    k_points = bandpath(kpts.build())
    assert len(k_points) == 10
    assert_equal(k_points[[0, 4, 9]], s_points)

    kpts.Nk = 50
    k_points = bandpath(kpts.build())
    assert len(k_points) == 50
    assert_equal(k_points[[0, 23, 49]], s_points)


def test_bandpath_xukg():
    """Test a simple band path in the FCC lattice that includes a jump between special points."""
    s_points = [
        SPECIAL_POINTS["fcc"]["X"],
        SPECIAL_POINTS["fcc"]["U"],
        SPECIAL_POINTS["fcc"]["K"],
        SPECIAL_POINTS["fcc"]["G"],
    ]

    kpts = KPoints("fcc", LATTICE_VECTORS["fcc"])
    kpts.path = "XU,KG"
    kpts.Nk = 4
    k_points = bandpath(kpts.build())
    assert len(k_points) == 4
    assert_equal(k_points, s_points)

    kpts.Nk = 10
    k_points = bandpath(kpts.build())
    assert len(k_points) == 10
    assert_equal(k_points[[0, 3, 4, 9]], s_points)

    kpts.Nk = 50
    k_points = bandpath(kpts.build())
    assert len(k_points) == 50
    assert_equal(k_points[[0, 13, 14, 49]], s_points)


def test_kpoints2axis_lgx():
    """Test the k-point axis calculation for a simple band path in the FCC lattice."""
    kpts = KPoints("fcc", LATTICE_VECTORS["fcc"])
    kpts.path = "LGX"
    kpts.Nk = 20
    kpts.build()
    k_axis, s_axis, labels = kpoints2axis(kpts)
    assert labels == ["L", "G", "X"]
    assert len(s_axis) == 3
    assert len(k_axis) == 20
    assert k_axis[0] == 0
    for s in s_axis:
        assert s in k_axis


def test_kpoints2axis_xukg():
    """Test the k-point axis calculation for a simple band path that includes a jump between."""
    kpts = KPoints("fcc", LATTICE_VECTORS["fcc"])
    kpts.path = "XU,KG"
    kpts.Nk = 25
    kpts.build()
    k_axis, s_axis, labels = kpoints2axis(kpts)
    assert labels == ["X", "U,K", "G"]
    assert len(s_axis) == 3
    assert len(k_axis) == 25
    assert k_axis[0] == 0
    assert k_axis[6] == k_axis[7]  # No distance between jumps
    for s in s_axis:
        assert s in k_axis


def test_get_brillouin_zone():
    """Test the Brillouin zone generation."""
    ridges = get_brillouin_zone(np.eye(3))
    # The Brillouin zone of a cubic lattice is cubic again
    assert_allclose(np.abs(ridges), np.pi)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
