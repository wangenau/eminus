#!/usr/bin/env python3
"""Test utility functions."""
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest
from scipy.special import sph_harm

from eminus.utils import add_maybe_none, pseudo_uniform, Ylm_real


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


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
