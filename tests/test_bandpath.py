#!/usr/bin/env python3
"""Test the bandpath sampler."""
import numpy as np
from numpy.testing import assert_allclose

from eminus.bandpath import bandpath, kpoints2axis
from eminus.data import SPECIAL_POINTS

FCC_LATTICE = np.array([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]])


def test_bandpath_lgx():
    """Test a simple band path in the FCC lattice."""
    s_points = [SPECIAL_POINTS['fcc']['L'],
                SPECIAL_POINTS['fcc']['G'],
                SPECIAL_POINTS['fcc']['X']]

    k_points = bandpath('fcc', FCC_LATTICE, 'LGX', 3)
    assert len(k_points) == 3
    assert_allclose(k_points, s_points)

    k_points = bandpath('fcc', FCC_LATTICE, 'LGX', 10)
    assert len(k_points) == 10
    assert_allclose(k_points[[0, 4, 9]], s_points)

    k_points = bandpath('fcc', FCC_LATTICE, 'LGX', 50)
    assert len(k_points) == 50
    assert_allclose(k_points[[0, 23, 49]], s_points)


def test_bandpath_xukg():
    """Test a simple band path in the FCC lattice that includes a jump between special points."""
    s_points = [SPECIAL_POINTS['fcc']['X'],
                SPECIAL_POINTS['fcc']['U'],
                SPECIAL_POINTS['fcc']['K'],
                SPECIAL_POINTS['fcc']['G']]

    k_points = bandpath('fcc', FCC_LATTICE, 'XU,KG', 4)
    assert len(k_points) == 4
    assert_allclose(k_points, s_points)

    k_points = bandpath('fcc', FCC_LATTICE, 'XU,KG', 10)
    assert len(k_points) == 10
    assert_allclose(k_points[[0, 2, 3, 9]], s_points)

    k_points = bandpath('fcc', FCC_LATTICE, 'XU,KG', 50)
    assert len(k_points) == 50
    assert_allclose(k_points[[0, 12, 13, 49]], s_points)


def test_kpoints2axis_lgx():
    """Test the k-point axis calculation for a simple band path in the FCC lattice."""
    k_points = bandpath('fcc', FCC_LATTICE, 'LGX', 20)
    k_axis, s_axis, labels = kpoints2axis('fcc', FCC_LATTICE, 'lgx', k_points)
    assert labels == ['L', 'G', 'X']
    assert len(s_axis) == 3
    assert len(k_axis) == 20
    assert k_axis[0] == 0
    for s in s_axis:
        assert s in k_axis


def test_kpoints2axis_xukg():
    """Test the k-point axis calculation for a simple band path that includes a jump between."""
    k_points = bandpath('fcc', FCC_LATTICE, 'XU,KG', 25)
    k_axis, s_axis, labels = kpoints2axis('fcc', FCC_LATTICE, 'xu,kg', k_points)
    assert labels == ['X', 'U,K', 'G']
    assert len(s_axis) == 3
    assert len(k_axis) == 25
    assert k_axis[0] == 0
    assert k_axis[6] == k_axis[7]  # No distance between jumps
    for s in s_axis:
        assert s in k_axis
