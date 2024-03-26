#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Test k-point symmetrization."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import norm

from eminus import Cell, SCF


@pytest.mark.parametrize('space_group', [True, False])
@pytest.mark.parametrize('time_reversal', [True, False])
def test_symmetrize(space_group, time_reversal):
    """Test the symmetrization of k-points."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import symmetrize

    cell = Cell('Si', 'diamond', 1, 10, kmesh=3).build()
    orig_k = cell.kpts.k
    orig_wk = cell.kpts.wk
    symmetrize(cell, space_group=space_group, time_reversal=time_reversal)
    symm_k = cell.kpts.k
    symm_wk = cell.kpts.wk
    if space_group or time_reversal:
        assert len(orig_k) > len(symm_k)
        assert len(orig_wk) > len(symm_wk)
    else:
        # If no symmetrization option is set the k-points should not change
        assert len(orig_k) == len(symm_k)
        assert len(orig_wk) == len(symm_wk)
    assert_allclose(np.sum(cell.kpts.wk), 1)
    # Make sure the original k-points are included in the symmetrized ones
    for k in symm_k:
        assert_allclose(np.sort(norm(orig_k - k, axis=1))[0], 0, atol=1e-15)


def test_unbuilt():
    """Test unbuilt KPoints objects."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import symmetrize

    cell = Cell('Si', 'diamond', 1, 10, kmesh=3).build()
    symmetrize(cell)
    orig_k = cell.kpts.k
    cell.kpts.is_built = False
    symmetrize(cell)
    symm_k = cell.kpts.k
    assert_allclose(orig_k, symm_k)


def test_energies():
    """Compare the energies of normal and symmetrized k-meshes."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import symmetrize

    cell = Cell('Si', 'diamond', ecut=5, a=10.2631, kmesh=(3, 2, 1)).build()
    orig_k = cell.kpts.k
    scf = SCF(cell)
    orig_etot = scf.run()

    symmetrize(cell, space_group=False, time_reversal=True)
    symm_k = cell.kpts.k
    scf = SCF(cell)
    symm_etot = scf.run()

    assert len(orig_k) > len(symm_k)
    assert_allclose(orig_etot, symm_etot, atol=1e-7)


if __name__ == '__main__':
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
