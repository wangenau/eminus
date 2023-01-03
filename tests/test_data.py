#!/usr/bin/env python3
'''Test data availability.'''
import inspect
import pathlib

from numpy.testing import assert_equal
import pytest

from eminus.data import COVALENT_RADII, CPK_COLORS, SYMBOL2NUMBER
import eminus.pade


def test_data():
    '''Check that every data dictionary has all necessary keys.'''
    assert_equal(SYMBOL2NUMBER.keys(), COVALENT_RADII.keys())
    assert_equal(SYMBOL2NUMBER.keys(), CPK_COLORS.keys())


def test_gth_data():
    '''Check that every atom has at least one pseudopotential file.'''
    psp_path = pathlib.Path(inspect.getfile(eminus.pade)).parent
    for atom in SYMBOL2NUMBER.keys():
        if atom == 'X':
            continue
        f_psp = sorted(psp_path.glob(f'{atom}-q*'))
        assert len(f_psp) > 0


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
