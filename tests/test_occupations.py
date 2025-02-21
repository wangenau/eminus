# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the Occupations object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from eminus.occupations import Occupations


@pytest.mark.parametrize(("Nelec", "Nspin", "ref"), [(1, None, 2), (2, None, 1), (2, 2, 2)])
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
    assert_equal(occ.f, np.array([[[1], [0]]]))
    occ.Nspin = 1
    occ.fill()
    assert_equal(occ.f, np.array([[[1]]]))


@pytest.mark.parametrize(
    ("Nelec", "Nspin", "spin", "ref"),
    [(1, 1, None, 0), (1, 2, None, 1), (2, 2, None, 0), (2, 2, 2, 2), (1, 2, 1, 1), (1, 1, 2, 0)],
)
def test_spin(Nelec, Nspin, spin, ref):
    """Test the setting of spin."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.spin = spin
    occ.Nspin = Nspin
    assert occ.spin == ref
    assert occ.multiplicity == occ.spin + 1


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


@pytest.mark.parametrize(
    ("Nelec", "Nspin", "spin", "charge", "ref"),
    [
        (1, 1, 1, 0, np.array([[[1]]])),
        (2, 1, 0, 0, np.array([[[2]]])),
        (1, 2, 1, 0, np.array([[[1], [0]]])),
        (2, 2, 0, 0, np.array([[[1], [1]]])),
        (2, 2, 2, 0, np.array([[[1, 1], [0, 0]]])),
        (3, 1, 0, 0, np.array([[[2, 1]]])),
        (3, 2, 1, 0, np.array([[[1, 1], [1, 0]]])),
        (3, 2, 3, 0, np.array([[[1, 1, 1], [0, 0, 0]]])),
        (3, 2, 0, -1, np.array([[[1, 1], [1, 1]]])),
        (3, 2, 2, -1, np.array([[[1, 1, 1], [1, 0, 0]]])),
        (3, 2, 4, -1, np.array([[[1, 1, 1, 1], [0, 0, 0, 0]]])),
        (3, 2, 0, 1, np.array([[[1], [1]]])),
        (3, 2, 2, 1, np.array([[[1, 1], [0, 0]]])),
        (3, 1, 0, 3, np.array([[[0]]])),
        (3, 2, 0, 3, np.array([[[0], [0]]])),
        (3, 2, 0, 0, np.array([[[1, 0.5], [1, 0.5]]])),
        (3, 2, 2, 0, np.array([[[1, 1, 0.5], [0.5, 0, 0]]])),
    ],
)
def test_fill(Nelec, Nspin, spin, charge, ref):
    """Test the fill function."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.Nspin = Nspin
    occ.spin = spin
    occ.charge = charge
    occ.fill()
    assert_equal(occ.f, ref)


@pytest.mark.parametrize(
    ("f", "Nelec", "Nspin", "spin", "ref"),
    [
        (2, 2, 1, 3, np.array([[[2]]])),
        (2, 2, 2, 2, np.array([[[2], [0]]])),
        (1, 2, 2, 0, np.array([[[1], [1]]])),
        (1, 3, 2, 0, np.array([[[1, 0.5], [1, 0.5]]])),
        (0.5, 2, 2, 1, np.array([[[0.5, 0.5, 0.5], [0.5, 0, 0]]])),
        (2 / 3, 2, 1, 0, np.array([[[2 / 3, 2 / 3, 2 / 3]]])),
        (2 / 3, 3, 2, 1, np.array([[[2 / 3, 2 / 3, 2 / 3], [2 / 3, 1 / 3, 0]]])),
        (2 / 3, 3, 2, 2, np.array([[[2 / 3, 2 / 3, 2 / 3, 0.5], [0.5, 0, 0, 0]]])),
    ],
)
def test_f_change(f, Nelec, Nspin, spin, ref):
    """Test changing f with a number."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.Nspin = Nspin
    occ.spin = spin
    occ.f = f
    assert_allclose(occ.f, ref)


@pytest.mark.parametrize(
    ("f", "Nelec_init", "Nelec", "Nspin", "Nstate", "spin", "charge"),
    [
        (np.array([3]), 3, 3, 1, 1, 0, 0),
        (np.array([2, 2]), 3, 4, 1, 2, 0, -1),
        (np.array([[1, 1], [1, 0]]), 3, 3, 2, 2, 1, 0),
        (np.array([[1, 1], [0, 0]]), 3, 2, 2, 2, 2, 1),
        (np.array([[1, 0.5], [1, 0.5]]), 3, 3, 2, 2, 0, 0),
        (np.array([[1, 1 / 3, 0, 2 / 3], [2, 0.5, 0.75, 0.75]]), 7, 6, 2, 4, 2, 1),
    ],
)
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
    occ.f = np.array([[[1, 2, 3], [0, 1, 2]]])
    assert_equal(occ.F[0][0], np.diag([1, 2, 3]))
    assert_equal(occ.F[0][1], np.diag([0, 1, 2]))


def test_Nk():
    """Test the Nk property."""
    occ = Occupations()
    assert occ.Nk == 1
    occ.Nk = 2
    assert occ.Nk == 2


def test_wk():
    """Test the wk property."""
    occ = Occupations()
    occ.wk = [1, 1]
    assert_equal(occ.wk, 1)
    assert occ.Nk == 2


@pytest.mark.parametrize(
    ("Nelec", "Nspin", "spin", "bands", "ref"),
    [
        (2, 1, 0, 2, np.array([[[2, 0]]])),
        (2, 2, 0, 2, np.array([[[1, 0], [1, 0]]])),
        (4, 1, 0, 4, np.array([[[2, 2, 0, 0]]])),
        (4, 2, 2, 4, np.array([[[1, 1, 1, 0], [1, 0, 0, 0]]])),
        (4, 2, 1, 4, np.array([[[1, 1, 0.5, 0], [1, 0.5, 0, 0]]])),
        (4, 2, 3, 5, np.array([[[1, 1, 1, 0.5, 0], [0.5, 0, 0, 0, 0]]])),
    ],
)
def test_bands(Nelec, Nspin, spin, bands, ref):
    """Test the bands property."""
    occ = Occupations()
    occ.Nelec = Nelec
    occ.Nspin = Nspin
    occ.spin = spin
    occ.smearing = 1
    occ.bands = bands
    occ.fill()
    assert_allclose(occ.f, ref)


@pytest.mark.parametrize("Nspin", [1, 2])
def test_smearing(Nspin):
    """Test the smearing property."""
    occ = Occupations()
    occ.Nelec = 2
    occ.Nspin = Nspin
    occ.spin = 1
    occ.fill()
    occ.smearing = 1
    assert not occ.is_filled
    occ.fill()
    assert occ.Nempty == 0


@pytest.mark.parametrize(
    ("Nspin", "spin", "wk", "ref"),
    [
        (1, 0, [1], 0),
        (2, 0, [1], 0),
        (2, 1, [1], 0.5),
        (2, 2, [1], 1),
        (2, 2, [0.5, 0.5], 1),
    ],
)
def test_magnetization(Nspin, spin, wk, ref):
    """Test the magnetization property."""
    occ = Occupations()
    occ.Nelec = 2
    occ.Nspin = Nspin
    occ.spin = spin
    occ.wk = wk
    occ.fill()
    assert_allclose(occ.magnetization, ref)


def test_unfilled_magnetization():
    """Test the magnetization property for an unfilled object."""
    occ = Occupations()
    occ.Nelec = 2
    occ.Nspin = 2
    assert occ.magnetization == 0
    occ.wk = [1]
    occ.fill()
    occ.magnetization = 0
    assert occ.magnetization == 0


@pytest.mark.parametrize(
    ("charge", "bands", "magnetization", "ref"),
    [
        (0, 1, 0, [[[1], [1]]]),
        (0, 2, 1, [[[1, 1], [0, 0]]]),
        (0, 2, -1, [[[0, 0], [1, 1]]]),
        (0, 2, 0, [[[1, 0], [1, 0]]]),
        (0, 3, 0.5, [[[1, 0.5, 0], [0.5, 0, 0]]]),
    ],
)
def test_magnetization_setter(charge, bands, magnetization, ref):
    """Test the magnetization setter."""
    occ = Occupations()
    occ.Nelec = 2
    occ.Nspin = 2
    occ.charge = charge
    occ.wk = [1]
    occ.bands = bands
    occ.smearing = 1
    occ.magnetization = magnetization
    assert_allclose(occ.f, ref)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
