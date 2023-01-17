#!/usr/bin/env python3
'''Test utility functions.'''
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus.utils import pseudo_uniform


@pytest.mark.parametrize('seed, ref', [(1234, np.array([[[0.93006472, 0.15416989, 0.93472344]]])),
                                       (42, np.array([[[0.57138534, 0.34186435, 0.13408117]]]))])
def test_pseudo_uniform(seed, ref):
    '''Test the reproduciblity of the pseudo random number generator.'''
    out = pseudo_uniform((1, 1, 3), seed=seed)
    assert_allclose(out, ref)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
