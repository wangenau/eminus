# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test operator identities."""

import typing  # noqa: F401

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from eminus import Atoms, config

config.backend = "scipy"

# Create an Atoms object to build mock wave functions
atoms = Atoms("Ne", (0, 0, 0), ecut=1).build()
rng = default_rng()
W_tests = {
    "full": rng.standard_normal((len(atoms.G2), atoms.occ.Nstate)),
    "active": rng.standard_normal((len(atoms.G2c), atoms.occ.Nstate)),
    "full_single": rng.standard_normal(len(atoms.G2)),
    "active_single": rng.standard_normal(len(atoms.G2c)),
    "full_spin": rng.standard_normal((atoms.occ.Nspin, len(atoms.G2), atoms.occ.Nstate)),
    "active_spin": rng.standard_normal((atoms.occ.Nspin, len(atoms.G2c), atoms.occ.Nstate)),
    "full_k": [rng.standard_normal((atoms.occ.Nspin, len(atoms.G2), atoms.occ.Nstate))],
    "active_k": [rng.standard_normal((atoms.occ.Nspin, len(atoms.Gk2c[0]), atoms.occ.Nstate))],
}  # type: dict[str, typing.Any]
dr = rng.standard_normal(3)


@pytest.mark.parametrize("field", ["full", "full_spin"])
def test_LinvL(field):
    """Test Laplacian operator identity."""
    out = atoms.Linv(atoms.L(W_tests[field]))
    test = np.copy(W_tests[field])
    if test.ndim == 3:
        test[:, 0, :] = 0
    else:
        test[0, :] = 0
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full", "full_spin"])
def test_LLinv(field):
    """Test Laplacian operator identity."""
    out = atoms.L(atoms.Linv(W_tests[field]))
    test = np.copy(W_tests[field])
    if test.ndim == 3:
        test[:, 0, :] = 0
    else:
        test[0, :] = 0
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full", "full_single", "full_spin", "full_k"])
def test_IJ(field):
    """Test forward and backward operator identity."""
    out = atoms.I(atoms.J(W_tests[field]))
    out = atoms.I(atoms.J(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize(
    "field",
    [
        "full",
        "active",
        "full_single",
        "active_single",
        "full_spin",
        "active_spin",
        "full_k",
        "active_k",
    ],
)
def test_JI(field):
    """Test forward and backward operator identity."""
    if "active" in field:
        out = atoms.J(atoms.I(W_tests[field]), full=False)
    else:
        out = atoms.J(atoms.I(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["active", "active_single", "active_spin", "active_k"])
def test_IdagJdag(field):
    """Test daggered forward and backward operator identity."""
    out = atoms.Idag(atoms.Jdag(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full", "full_single", "full_spin", "full_k"])
def test_JdagIdag(field):
    """Test daggered forward and backward operator identity."""
    out = atoms.Jdag(atoms.Idag(W_tests[field], full=True))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full_single"])
def test_hermitian_I(field):
    """Test that I and Idag operators are hermitian."""
    a = W_tests[field]
    b = W_tests[field] + rng.standard_normal(1)
    assert not isinstance(a, list)
    out = (a.conj().T @ atoms.I(b)).conj()
    test = b.conj().T @ atoms.Idag(a, full=True)
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full_single"])
def test_hermitian_J(field):
    """Test that J and Jdag operators are hermitian."""
    a = W_tests[field]
    b = W_tests[field] + rng.standard_normal(1)
    assert not isinstance(a, list)
    out = (a.conj().T @ atoms.J(b)).conj()
    test = b.conj().T @ atoms.Jdag(a)
    assert_allclose(out, test)


@pytest.mark.parametrize(
    "field",
    [
        "full",
        "active",
        "full_single",
        "active_single",
        "full_spin",
        "active_spin",
        "full_k",
        "active_k",
    ],
)
def test_TT(field):
    """Test translation operator identity."""
    out = atoms.T(atoms.T(W_tests[field], dr), -dr)
    test = W_tests[field]
    assert_allclose(out, test)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
