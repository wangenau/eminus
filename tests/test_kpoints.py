#!/usr/bin/env python3
"""Test the k-points functionalities."""
import numpy as np
from numpy.testing import assert_equal

from eminus.data import LATTICE_VECTORS, SPECIAL_POINTS
from eminus.kpoints import bandpath, KPoints, kpoints2axis, monkhorst_pack


def test_monkhorst_pack():
    """Test the Monkhorst-Pack mesh generation."""
    k_points = monkhorst_pack((1, 1, 1))
    assert_equal(k_points, 0)
    k_points = monkhorst_pack((2, 2, 2))
    assert_equal(np.abs(k_points), 1 / 4)


def test_bandpath_lgx():
    """Test a simple band path in the FCC lattice."""
    s_points = [SPECIAL_POINTS['fcc']['L'],
                SPECIAL_POINTS['fcc']['G'],
                SPECIAL_POINTS['fcc']['X']]

    kpts = KPoints('fcc', LATTICE_VECTORS['fcc'])
    kpts.path = 'LGX'
    kpts.Nk = 3
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
    s_points = [SPECIAL_POINTS['fcc']['X'],
                SPECIAL_POINTS['fcc']['U'],
                SPECIAL_POINTS['fcc']['K'],
                SPECIAL_POINTS['fcc']['G']]

    kpts = KPoints('fcc', LATTICE_VECTORS['fcc'])
    kpts.path = 'XU,KG'
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
    kpts = KPoints('fcc', LATTICE_VECTORS['fcc'])
    kpts.path = 'LGX'
    kpts.Nk = 20
    kpts.build()
    k_axis, s_axis, labels = kpoints2axis(kpts)
    assert labels == ['L', 'G', 'X']
    assert len(s_axis) == 3
    assert len(k_axis) == 20
    assert k_axis[0] == 0
    for s in s_axis:
        assert s in k_axis


def test_kpoints2axis_xukg():
    """Test the k-point axis calculation for a simple band path that includes a jump between."""
    kpts = KPoints('fcc', LATTICE_VECTORS['fcc'])
    kpts.path = 'XU,KG'
    kpts.Nk = 25
    kpts.build()
    k_axis, s_axis, labels = kpoints2axis(kpts)
    assert labels == ['X', 'U,K', 'G']
    assert len(s_axis) == 3
    assert len(k_axis) == 25
    assert k_axis[0] == 0
    assert k_axis[6] == k_axis[7]  # No distance between jumps
    for s in s_axis:
        assert s in k_axis


if __name__ == '__main__':
    import inspect
    import pathlib

    import pytest
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
