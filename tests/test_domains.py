#!/usr/bin/env python3
'''Test domain generation.'''
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms
from eminus.domains import domain_cuboid, domain_sphere, truncate


atoms = Atoms('CH4', [[0, 0, 0],
                      [1.18599212, 1.18599212, 1.18599212],
                      [1.18599212, -1.18599212, -1.18599212],
                      [-1.18599212, 1.18599212, -1.18599212],
                      [-1.18599212, -1.18599212, 1.18599212]]).build()


@pytest.mark.parametrize('length', [0.001, 0.01, 0.1, 1, 10])
def test_domain_cuboid(length):
    '''Test cuboidal domain generation by ensuring that the grid points are close to the center.'''
    out = truncate(atoms.r, domain_cuboid(atoms, length))
    ref = np.zeros_like(out)
    assert_allclose(out, ref, atol=length)


@pytest.mark.parametrize('radius', [0.001, 0.01, 0.1, 1, 10])
def test_domain_sphere(radius):
    '''Test sperical domain generation by ensuring that the grid points are close to the center.'''
    out = truncate(atoms.r, domain_sphere(atoms, radius))
    ref = np.zeros_like(out)
    assert_allclose(out, ref, atol=radius)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
