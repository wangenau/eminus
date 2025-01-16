# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test torch extra."""

import typing  # noqa: F401

import numpy as np  # noqa: F401
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from eminus import Atoms, config

config.backend = "torch"
print(config.backend)  # Call the getter function once

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


@pytest.mark.parametrize("field", ["full", "full_single", "full_spin", "full_k"])
def test_IJ(field):
    """Test forward and backward operator identity."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
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
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    if "active" in field:
        out = atoms.J(atoms.I(W_tests[field]), full=False)
    else:
        out = atoms.J(atoms.I(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["active", "active_single", "active_spin", "active_k"])
def test_IdagJdag(field):
    """Test daggered forward and backward operator identity."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    out = atoms.Idag(atoms.Jdag(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full", "full_single", "full_spin", "full_k"])
def test_JdagIdag(field):
    """Test daggered forward and backward operator identity."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    out = atoms.Jdag(atoms.Idag(W_tests[field], full=True))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full_single"])
def test_hermitian_I(field):
    """Test that I and Idag operators are hermitian."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    a = W_tests[field]
    b = W_tests[field] + rng.standard_normal(1)
    assert not isinstance(a, list)
    out = (a.conj().T @ atoms.I(b)).conj()
    test = b.conj().T @ atoms.Idag(a, full=True)
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full_single"])
def test_hermitian_J(field):
    """Test that J and Jdag operators are hermitian."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    a = W_tests[field]
    b = W_tests[field] + rng.standard_normal(1)
    assert not isinstance(a, list)
    out = (a.conj().T @ atoms.J(b)).conj()
    test = b.conj().T @ atoms.Jdag(a)
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full", "full_single", "full_spin", "full_k"])
def test_IJ_gpu(field):
    """Test forward and backward GPU operator identity."""
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip("GPU not available, skip tests")
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
def test_JI_gpu(field):
    """Test forward and backward GPU operator identity."""
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip("GPU not available, skip tests")
    if "active" in field:
        out = atoms.J(atoms.I(W_tests[field]), full=False)
    else:
        out = atoms.J(atoms.I(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["active", "active_single", "active_spin", "active_k"])
def test_IdagJdag_gpu(field):
    """Test daggered forward and backward GPU operator identity."""
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip("GPU not available, skip tests")
    out = atoms.Idag(atoms.Jdag(W_tests[field]))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full", "full_single", "full_spin", "full_k"])
def test_JdagIdag_gpu(field):
    """Test daggered forward and backward GPU operator identity."""
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip("GPU not available, skip tests")
    out = atoms.Jdag(atoms.Idag(W_tests[field], full=True))
    test = W_tests[field]
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full_single"])
def test_hermitian_I_gpu(field):
    """Test that I and Idag GPU operators are hermitian."""
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip("GPU not available, skip tests")
    a = W_tests[field]
    b = W_tests[field] + rng.standard_normal(1)
    assert not isinstance(a, list)
    out = (a.conj().T @ atoms.I(b)).conj()
    test = b.conj().T @ atoms.Idag(a, full=True)
    assert_allclose(out, test)


@pytest.mark.parametrize("field", ["full_single"])
def test_hermitian_J_gpu(field):
    """Test that J and Jdag GPU operators are hermitian."""
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip("GPU not available, skip tests")
    a = W_tests[field]
    b = W_tests[field] + rng.standard_normal(1)
    assert not isinstance(a, list)
    out = (a.conj().T @ atoms.J(b)).conj()
    test = b.conj().T @ atoms.Jdag(a)
    assert_allclose(out, test)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
