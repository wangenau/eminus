#!/usr/bin/env python3
"""Test the k-points functionalities."""
import numpy as np
from numpy.testing import assert_equal

from eminus.data import LATTICE_VECTORS, SPECIAL_POINTS
from eminus.kpoints import bandpath, kpoints2axis, monkhorst_pack


def test_monkhorst_pack():
    """Test the Monkhorst-Pack mesh generation."""
    k_points, wk = monkhorst_pack((1, 1, 1), LATTICE_VECTORS['sc'])
    assert_equal(wk, 1)
    assert_equal(k_points, 0)
    k_points, wk = monkhorst_pack((2, 2, 2), LATTICE_VECTORS['sc'])
    assert_equal(wk, 1 / 8)
    assert_equal(np.abs(k_points), np.pi / 2)


def test_bandpath_lgx():
    """Test a simple band path in the FCC lattice."""
    s_points = [SPECIAL_POINTS['fcc']['L'],
                SPECIAL_POINTS['fcc']['G'],
                SPECIAL_POINTS['fcc']['X']]

    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'LGX', 3)
    assert len(k_points) == 3
    assert_equal(k_points, s_points)

    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'LGX', 10)
    assert len(k_points) == 10
    assert_equal(k_points[[0, 4, 9]], s_points)

    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'LGX', 50)
    assert len(k_points) == 50
    assert_equal(k_points[[0, 23, 49]], s_points)


def test_bandpath_xukg():
    """Test a simple band path in the FCC lattice that includes a jump between special points."""
    s_points = [SPECIAL_POINTS['fcc']['X'],
                SPECIAL_POINTS['fcc']['U'],
                SPECIAL_POINTS['fcc']['K'],
                SPECIAL_POINTS['fcc']['G']]

    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'XU,KG', 4)
    assert len(k_points) == 4
    assert_equal(k_points, s_points)

    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'XU,KG', 10)
    assert len(k_points) == 10
    assert_equal(k_points[[0, 2, 3, 9]], s_points)

    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'XU,KG', 50)
    assert len(k_points) == 50
    assert_equal(k_points[[0, 12, 13, 49]], s_points)


def test_kpoints2axis_lgx():
    """Test the k-point axis calculation for a simple band path in the FCC lattice."""
    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'LGX', 20)
    k_axis, s_axis, labels = kpoints2axis('fcc', LATTICE_VECTORS['fcc'], 'lgx', k_points)
    assert labels == ['L', 'G', 'X']
    assert len(s_axis) == 3
    assert len(k_axis) == 20
    assert k_axis[0] == 0
    for s in s_axis:
        assert s in k_axis


def test_kpoints2axis_xukg():
    """Test the k-point axis calculation for a simple band path that includes a jump between."""
    k_points = bandpath('fcc', LATTICE_VECTORS['fcc'], 'XU,KG', 25)
    k_axis, s_axis, labels = kpoints2axis('fcc', LATTICE_VECTORS['fcc'], 'xu,kg', k_points)
    assert labels == ['X', 'U,K', 'G']
    assert len(s_axis) == 3
    assert len(k_axis) == 25
    assert k_axis[0] == 0
    assert k_axis[6] == k_axis[7]  # No distance between jumps
    for s in s_axis:
        assert s in k_axis
