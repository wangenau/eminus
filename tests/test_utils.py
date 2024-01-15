#!/usr/bin/env python3
"""Test utility functions."""
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.special import sph_harm

from eminus import Atoms, config
from eminus.utils import (
    add_maybe_none,
    atom2charge,
    get_lattice,
    handle_k_gracefully,
    handle_k_indexable,
    handle_k_reducable,
    handle_spin_gracefully,
    handle_torch,
    molecule2list,
    pseudo_uniform,
    skip_k,
    vector_angle,
    Ylm_real,
)


@pytest.mark.parametrize('l', [0, 1, 2, 3])
def test_Ylm(l):
    """Test the spherical harmonics."""
    # Generate random G
    # Somehow I can only get the correct results from scipy when using positive G
    rng = default_rng()
    G = np.abs(rng.random((1000, 3)))

    # Calculate the spherical coordinates theta and phi
    tmp = np.sqrt(G[:, 0]**2 + G[:, 1]**2) / G[:, 2]
    theta = np.arctan(tmp)
    theta[G[:, 0] < 0] += np.pi
    theta[np.abs(G[:, 0]) == 0] = np.pi / 2

    phi = np.arctan2(G[:, 1], G[:, 0])
    phi_idx = (np.abs(G[:, 0]) == 0)
    phi[phi_idx] = np.pi / 2 * np.sign(G[phi_idx, 1])

    # Calculate the spherical harmonics
    for m in range(-l, l + 1):
        Y_intern = Ylm_real(l, m, G)
        Y_extern = sph_harm(abs(m), l, phi, theta)
        if m < 0:
            Y_extern = np.sqrt(2) * (-1)**m * Y_extern.imag
        elif m > 0:
            Y_extern = np.sqrt(2) * (-1)**m * Y_extern.real
        assert_allclose(Y_intern, Y_extern)


@pytest.mark.parametrize(('seed', 'ref'), [
    (1234, np.array([[[0.93006472, 0.15416989, 0.93472344]]])),
    (42, np.array([[[0.57138534, 0.34186435, 0.13408117]]]))])
def test_pseudo_uniform(seed, ref):
    """Test the reproduciblity of the pseudo random number generator."""
    out = pseudo_uniform((1, 1, 3), seed=seed)
    assert_allclose(out, ref)


@pytest.mark.parametrize(('a', 'b', 'ref'), [(1, 2, 3),
                                             (1, None, 1),
                                             (None, 2, 2),
                                             (None, None, None)])
def test_add_maybe_none(a, b, ref):
    """Test the function to add two variables that can be None."""
    out = add_maybe_none(a, b)
    assert out == ref


@pytest.mark.parametrize(('molecule', 'ref'), [('CH4', ['C', 'H', 'H', 'H', 'H']),
                                               ('HeX', ['He', 'X']),
                                               ('CH2O2', ['C', 'H', 'H', 'O', 'O'])])
def test_molecule2list(molecule, ref):
    """Test the molecule to list expansion."""
    out = molecule2list(molecule)
    assert out == ref


@pytest.mark.parametrize(('atom', 'ref'), [(['H'], [1]),
                                           (['Li'], [3]),
                                           (['He', 'He'], [2, 2]),
                                           (['C', 'H', 'H', 'H', 'H'], [4, 1, 1, 1, 1])])
def test_atom2charge(atom, ref):
    """Test the molecule to charge expansion."""
    out = atom2charge(atom)
    assert out == ref


@pytest.mark.parametrize(('a', 'b', 'ref'), [([1, 0], [0, 1], 90),
                                             ([1, 0, 0], [0, 1, 0], 90),
                                             ([1, 1, 0], [0, 1, 1], 60),
                                             ([3, -2], [1, 7], 115.559965)])
def test_vector_angle(a, b, ref):
    """Test the vector angle calculation."""
    out = vector_angle(a, b)
    assert_allclose(out, ref, )


def test_get_lattice():
    """Test the lattice utility function."""
    out = get_lattice(np.eye(3))
    for vert in out:
        assert_equal(np.linalg.norm(vert[0] - vert[1]), 1)
    out = get_lattice(np.ones((3, 3)) - np.eye(3))
    for vert in out:
        assert_equal(np.linalg.norm(vert[0] - vert[1]), np.sqrt(2))


def test_handle_spin_gracefully():
    """Test the test_handle_spin_gracefully decorator."""
    @handle_spin_gracefully
    def mock(obj, W):
        return W
    W = np.ones((1, 1, 1))
    out = mock(None, W)
    assert_equal(out, W)
    out = mock(None, W[0])
    assert_equal(out, W[0])


def test_skip_k():
    """Test the skip_k decorator."""
    @skip_k
    def mock(obj, W):
        assert isinstance(W, np.ndarray)
        return W
    atoms = Atoms('He', (0, 0, 0))
    W = [np.ones((1, 1, 1))]
    out = mock(atoms, W)
    assert_equal(out, W)
    out = mock(atoms, W[0])
    assert_equal(out, W[0])
    out = mock(atoms, W[0][0])
    assert_equal(out, W[0][0])


def test_handle_k_gracefully():
    """Test the handle_k_gracefully decorator."""
    @handle_k_gracefully
    def mock(obj, W):
        return W

    def mock_val(obj, W):
        return 0
    W = [np.ones((1, 1, 1))]
    out = mock(None, W)
    assert_equal(out, W)
    out = mock(None, W[0])
    assert_equal(out, W[0])
    out = mock_val(None, W)
    assert_equal(out, 0)


def test_handle_k_indexable():
    """Test the handle_k_indexable decorator."""
    @handle_k_indexable
    def mock(obj, W, ik=0):
        return W
    W = [np.ones((1, 1, 1))] * 2
    out = mock(None, W)
    assert_equal(out, W)
    out = mock(None, W[0])
    assert_equal(out, W[0])


def test_handle_k_reducable():
    """Test the handle_k_reducable decorator."""
    @handle_k_reducable
    def mock(obj, W, ik=0):
        return W
    W = [np.ones((1, 1, 1))] * 2
    out = mock(None, W)
    assert_equal(out, np.ones((1, 1, 1)) * 2)
    out = mock(None, W[0])
    assert_equal(out, W[0])


def test_handle_torch():
    """Test the handle_torch decorator."""
    @handle_torch
    def mock(x):
        return x
    config.use_torch = False
    out = mock(np.pi)
    assert_equal(out, np.pi)
    config.use_torch = True
    with pytest.raises(AttributeError):
        mock(np.pi)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
