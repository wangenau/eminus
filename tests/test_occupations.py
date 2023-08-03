#!/usr/bin/env python3
"""Test the Occupations object."""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from eminus.occupations import Occupations


@pytest.mark.parametrize(('Nelec', 'Nspin', 'ref'), [(1, None, 2),
                                                     (2, None, 1),
                                                     (2, 2, 2)])
def test_Nspin(Nelec, Nspin, ref):
    """Test the setting of Nspin."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.Nspin = Nspin
    assert occ.Nspin == ref


def test_Nspin_change():
    """Test changing Nspin."""
    occ = Occupations()
    occ.Nelec = 1
    occ.Nspin = 2
    occ.spin = 1
    occ.Nspin = 1
    assert occ.spin == 0


def test_Nspin_change_after_fill():
    """Test changing Nspin after the object was filled."""
    occ = Occupations()
    occ.Nelec = 1
    occ.Nspin = 2
    occ.spin = 1
    occ.fill()
    assert_equal(occ.f, np.array([[1], [0]]))
    occ.Nspin = 1
    occ.fill()
    assert_equal(occ.f, np.array([[1]]))


@pytest.mark.parametrize(('Nelec', 'Nspin', 'spin', 'ref'), [(1, 1, None, 0),
                                                             (1, 2, None, 1),
                                                             (2, 2, None, 0),
                                                             (2, 2, 2, 2),
                                                             (1, 2, 1, 1),
                                                             (1, 1, 2, 0)])
def test_spin(Nelec, Nspin, spin, ref):
    """Test the setting of spin."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.spin = spin
    occ.Nspin = Nspin
    assert occ.spin == ref


def test_charge_change():
    """Test changing the charge."""
    occ = Occupations()
    occ.charge = 1
    assert occ.charge == 1
    occ.Nelec = 10
    occ.charge = -1
    assert occ.Nelec == 12

    occ = Occupations()
    occ.Nelec = 10
    occ.charge = 3
    assert occ.Nelec == 7


@pytest.mark.parametrize(('Nelec', 'Nspin', 'spin', 'charge', 'ref'),
                         [(1, 1, 1, 0, np.array([[1]])),
                          (2, 1, 0, 0, np.array([[2]])),
                          (1, 2, 1, 0, np.array([[1], [0]])),
                          (2, 2, 0, 0, np.array([[1], [1]])),
                          (2, 2, 2, 0, np.array([[1, 1], [0, 0]])),
                          (3, 1, 0, 0, np.array([[2, 1]])),
                          (3, 2, 1, 0, np.array([[1, 1], [1, 0]])),
                          (3, 2, 3, 0, np.array([[1, 1, 1], [0, 0, 0]])),
                          (3, 2, 0, -1, np.array([[1, 1], [1, 1]])),
                          (3, 2, 2, -1, np.array([[1, 1, 1], [1, 0, 0]])),
                          (3, 2, 4, -1, np.array([[1, 1, 1, 1], [0, 0, 0, 0]])),
                          (3, 2, 0, 1, np.array([[1], [1]])),
                          (3, 2, 2, 1, np.array([[1, 1], [0, 0]])),
                          (3, 1, 0, 3, np.array([[0]])),
                          (3, 2, 0, 3, np.array([[0], [0]])),
                          (3, 2, 0, 0, np.array([[1, 0.5], [1, 0.5]])),
                          (3, 2, 2, 0, np.array([[1, 1, 0.5], [0.5, 0, 0]]))])
def test_fill(Nelec, Nspin, spin, charge, ref):
    """Test the fill function."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.Nspin = Nspin
    occ.spin = spin
    occ.charge = charge
    occ.fill()
    assert_equal(occ.f, ref)


@pytest.mark.parametrize(('f', 'Nelec', 'Nspin', 'spin', 'ref'),
                         [(2, 2, 1, 3, np.array([[2]])),
                          (2, 2, 2, 2, np.array([[2], [0]])),
                          (1, 2, 2, 0, np.array([[1], [1]])),
                          (1, 3, 2, 0, np.array([[1, 0.5], [1, 0.5]])),
                          (0.5, 2, 2, 1, np.array([[0.5, 0.5, 0.5], [0.5, 0, 0]])),
                          (2 / 3, 2, 1, 0, np.array([[2 / 3, 2 / 3, 2 / 3]])),
                          (2 / 3, 3, 2, 1, np.array([[2 / 3, 2 / 3, 2 / 3], [2 / 3, 1 / 3, 0]])),
                          (2 / 3, 3, 2, 2, np.array([[2 / 3, 2 / 3, 2 / 3, 0.5], [0.5, 0, 0, 0]]))])
def test_f_change(f, Nelec, Nspin, spin, ref):
    """Test changing f with a number."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.Nspin = Nspin
    occ.spin = spin
    occ.f = f
    assert_allclose(occ.f, ref)


@pytest.mark.parametrize(('f', 'Nelec_init', 'Nelec', 'Nspin', 'Nstate', 'spin', 'charge'),
                         [(np.array([3]), 3, 3, 1, 1, 0, 0),
                          (np.array([2, 2]), 3, 4, 1, 2, 0, -1),
                          (np.array([[1, 1], [1, 0]]), 3, 3, 2, 2, 1, 0),
                          (np.array([[1, 1], [0, 0]]), 3, 2, 2, 2, 2, 1),
                          (np.array([[1, 0.5], [1, 0.5]]), 3, 3, 2, 2, 0, 0),
                          (np.array([[1, 1 / 3, 0, 2 / 3],
                                     [2, 0.5, 0.75, 0.75]]), 7, 6, 2, 4, 2, 1)])
def test_f_change_array(f, Nelec_init, Nelec, Nspin, Nstate, spin, charge):
    """Test changing f with an array."""
    occ = Occupations()
    occ.Nelec = Nelec_init
    occ.f = f
    assert occ.Nelec == Nelec
    assert occ.Nspin == Nspin
    assert occ.Nstate == Nstate
    assert occ.spin == spin
    assert occ.charge == charge


def test_F():
    """Test the F property."""
    occ = Occupations()
    occ.f = np.array([[1, 2, 3], [0, 1, 2]])
    assert_equal(occ.F[0], np.diag([1, 2, 3]))
    assert_equal(occ.F[1], np.diag([0, 1, 2]))


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
