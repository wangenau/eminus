# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the Atoms class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.linalg import det

from eminus import Atoms, Cell, log
from eminus.tools import center_of_mass

inp = ("He", (0, 0, 0))


@pytest.mark.parametrize(
    ("atom", "ref", "Nref"),
    [
        ("H", ["H"], 1),
        (["H"], ["H"], 1),
        ("He-q2", ["He"], 1),
        (["H", "He"], ["H", "He"], 2),
        ("CH4", ["C", "H", "H", "H", "H"], 5),
        ("HeX", ["He", "X"], 2),
    ],
)
def test_atom(atom, ref, Nref):
    """Test initialization of the atom variable."""
    atoms = Atoms(atom, [[0, 0, 0]] * Nref)
    assert atoms.atom == ref
    assert atoms.Natoms == Nref
    assert len(atoms.Z) == Nref


@pytest.mark.parametrize(
    ("pos", "center", "ref"),
    [
        ([[0, 0, 0]], None, [(0, 0, 0)]),
        ([[0, 0, 0]], True, [(10, 10, 10)]),
        ([[0, 0, 0]], "shift", [(10, 10, 10)]),
        ([[0] * 3, [1] * 3], "rotate", [(0, 0, 0), (np.sqrt(3), 0, 0)]),
        ([[0] * 3, [1] * 3], "shift", [[9.5] * 3, [10.5] * 3]),
        ([[0] * 3, [1] * 3], True, [[10 - np.sqrt(3) / 2, 10, 10], [10 + np.sqrt(3) / 2, 10, 10]]),
    ],
)
def test_pos(pos, center, ref):
    """Test the setting of the atom coordinates."""
    atoms = Atoms(["H"] * len(pos), pos=pos, center=center)
    assert_allclose(np.abs(atoms.pos), ref, atol=1e-15)


@pytest.mark.parametrize(
    ("a", "ref", "Omega"),
    [
        (5, [[5, 0, 0], [0, 5, 0], [0, 0, 5]], 125),
        ([2, 3, 4], [[2, 0, 0], [0, 3, 0], [0, 0, 4]], 24),
        ([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], [[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], 2),
    ],
)
def test_cell(a, ref, Omega):
    """Test the setting of cell size."""
    atoms = Atoms(*inp, a=a).build()
    assert atoms.Omega == Omega
    assert_equal(atoms.a, ref)
    assert_allclose(atoms.Omega, det(atoms.a))
    assert_equal(atoms.r[0], 0)
    assert len(atoms.r) == atoms.Ns
    assert_allclose(atoms.s[0] / atoms.s[1], abs(atoms.a[0, 0] / atoms.a[1, 1]), atol=0.1)
    assert atoms.dV == atoms.Omega / np.prod(atoms.s)
    assert atoms.is_built


@pytest.mark.parametrize("spin", [0, 1, 2])
def test_spin(spin):
    """Test the spin option."""
    atoms = Atoms(*inp, spin=spin)
    assert atoms.spin == spin
    assert atoms.occ.spin == spin


@pytest.mark.parametrize("charge", [0, 1, 2, -1, -2])
def test_charge(charge):
    """Test the charge option."""
    atoms = Atoms(*inp, charge=charge)
    assert atoms.charge == charge
    assert atoms.occ.charge == charge


@pytest.mark.parametrize(
    ("atom", "spin", "unrestricted", "ref"),
    [
        ("H", 0, None, 2),
        ("He", 0, None, 1),
        ("H", 0, True, 2),
        ("He", 0, False, 1),
        ("He", 1, None, 2),
        ("He", 2, None, 2),
        ("H", 0, False, 1),
    ],
)
def test_unrestricted(atom, spin, unrestricted, ref):
    """Test the spinpol option."""
    atoms = Atoms(atom=atom, pos=(0, 0, 0), spin=spin, unrestricted=unrestricted)
    assert atoms.occ.Nspin == ref


@pytest.mark.parametrize("center", [False, True, "rotate", "shift", "recentered"])
def test_center(center):
    """Test the center option."""
    atoms = Atoms("H2", [[0, 0, 0], [1, 1, 1]], center=center)
    if center is False:
        assert_equal(atoms.pos, [[0, 0, 0], [1, 1, 1]])
    elif center is True:
        assert_equal(atoms.pos, [[10 - np.sqrt(3) / 2, 10, 10], [10 + np.sqrt(3) / 2, 10, 10]])
    elif center == "rotate":
        assert_allclose(atoms.pos, [[0, 0, 0], [np.sqrt(3), 0, 0]], atol=1e-15)
    elif center == "shift":
        assert_equal(atoms.pos, [[9.5, 9.5, 9.5], [10.5, 10.5, 10.5]])
    elif center == "recentered":
        assert_equal(atoms.pos, [[0, 0, 0], [1, 1, 1]])


def test_verbose():
    """Test the verbosity level."""
    log.verbose = "DEBUG"
    atoms = Atoms(*inp)
    assert atoms.verbose == log.verbose
    level = "DEBUG"
    log.verbose = level
    atoms = Atoms(*inp)
    assert atoms.verbose == level
    atoms = Atoms(*inp, verbose=level)
    assert atoms.verbose == level


@pytest.mark.parametrize(
    ("f", "unrestricted", "fref", "Nref"),
    [
        (None, False, [[[2]]], 1),
        (None, True, [[[1], [1]]], 1),
        (2, False, [[[2]]], 1),
        (1, True, [[[1], [1]]], 1),
    ],
)
def test_f(f, unrestricted, fref, Nref):
    """Test the occupation and state options."""
    atoms = Atoms(*inp, unrestricted=unrestricted)
    atoms.s = 1
    atoms.f = f
    atoms.build()
    assert_equal(atoms.f, fref)
    assert atoms.occ.is_filled
    assert atoms.occ.Nstate == Nref


@pytest.mark.parametrize(("s", "ref"), [(2, [2] * 3), ([2, 3, 4], [2, 3, 4]), (None, 99)])
def test_s(s, ref):
    """Test the setting of sampling."""
    atoms = Atoms(*inp)
    if s is not None:
        atoms.s = s
    assert_equal(atoms.s, ref)


@pytest.mark.parametrize(
    ("atom", "Z", "ref", "Nref"),
    [
        ("H", None, [1], 1),
        ("Li", "pade", [1], 1),
        ("Li", "pbe", [3], 3),
        ("Li-q3", None, [3], 3),
        ("He2", 2, [2, 2], 4),
        ("CH4", {"C": 3, "H": 2}, [3, 2, 2, 2, 2], 11),
        ("Ne", 10, [10], 10),
    ],
)
def test_Z(atom, Z, ref, Nref):
    """Test setting of charges."""
    atoms = Atoms(atom, [[0, 0, 0]] * len(ref))
    atoms.Z = Z
    atoms.build()
    assert_equal(atoms.Z, ref)
    assert len(atoms.Z) == atoms.Natoms
    assert atoms.occ.Nelec == Nref
    assert atoms.occ.Nempty >= 0


def test_G():
    """Test the setting of G-vectors."""
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    assert_equal(atoms.G[0], 0)
    assert len(atoms.G) == atoms.Ns
    assert_equal(atoms.G2, atoms.G2c)
    assert_equal(atoms.Sf, 1)
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    assert len(atoms.G2) == len(atoms.G2c)


def test_Gk():
    """Test the setting of G+k-vectors."""
    atoms = Cell("He", "fcc", 30, 5)
    atoms.kpts.kmesh = (2, 1, 2)
    atoms.build()
    assert len(atoms.Gk2) == atoms.kpts.Nk + 1
    assert len(atoms.Gk2[0]) == len(atoms.G2)
    assert len(atoms.Gk2c) == atoms.kpts.Nk + 1
    assert len(atoms.Gk2c[0]) != len(atoms.Gk2c[1])


def test_kpts():
    """Test the k-points object."""
    atoms = Atoms(*inp, a=(1, 2, 3)).build()
    assert_equal(atoms.kpts.a, atoms.a)
    atoms.a = [2] * 3
    assert_equal(atoms.kpts.a, atoms.a)
    assert_equal(atoms.occ.wk, atoms.kpts.wk)
    atoms.kpts.wk = [0.5]
    atoms.kpts.is_built = True
    atoms.build()
    assert_equal(atoms.occ.wk, atoms.kpts.wk)


def test_operators():
    """Test that operators are properly set and callable."""
    atoms = Atoms(*inp).build()
    for op in ("O", "L", "Linv", "I", "J", "Idag", "Jdag", "K", "T"):
        assert callable(getattr(atoms, op))


def test_clear():
    """Test the clear function."""
    atoms = Atoms(*inp).build()
    assert atoms.is_built
    atoms.clear()
    assert not atoms.is_built
    assert [x for x in (atoms.r, atoms.G, atoms.G2, atoms.active, atoms.G2c, atoms.Sf) if x is None]


def test_recenter():
    """Test the recenter function."""
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    Sf_old = atoms.Sf
    center = (1, 1, 1)
    atoms.recenter(center)
    assert_allclose(center_of_mass(atoms.pos), center)
    assert not np.array_equal(Sf_old, atoms.Sf)
    assert atoms.center == "recentered"


def test_set_k():
    """Test the setting of custom k-points."""
    atoms = Atoms(*inp)
    atoms.s = 2
    atoms.build()
    atoms.set_k([[0] * 3] * 2)
    assert atoms.kpts.Nk == 2
    assert len(atoms.kpts.k) == 2
    assert len(atoms.kpts.wk) == 2
    assert len(atoms.occ.wk) == 2
    assert len(atoms.Gk2) == 2 + 1
    assert len(atoms.Gk2c) == 2 + 1


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
