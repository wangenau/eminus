#!/usr/bin/env python3
"""Test fods identities."""

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_equal
import pytest

from eminus import Atoms
from eminus.extras import remove_core_fods, split_fods

rng = default_rng()


@pytest.mark.parametrize('unrestricted', [True, False])
@pytest.mark.parametrize(
    ('basis', 'loc'), [('pc-1', 'fb'), ('pc-0', 'er'), ('pc-0', 'pm'), ('pc-0', 'gpm')]
)
def test_get_fods_unpol(unrestricted, basis, loc):
    """Test FOD generator."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import get_fods

    atoms = Atoms('He', (0, 0, 0), unrestricted=unrestricted).build()
    fods = get_fods(atoms, basis=basis, loc=loc)
    # For He all FODs are core FODs and therefore should be close to the atom
    assert_allclose(atoms.pos, fods[0], atol=1e-6)


@pytest.mark.parametrize('unrestricted', [True, False])
@pytest.mark.parametrize('elec_symbols', [('X', 'He'), ('He', 'Ne')])
def test_split_fods(unrestricted, elec_symbols):
    """Test splitting FODs from atoms."""
    pos = rng.standard_normal((5, 3))
    atom = ['H'] * len(pos)
    fods = rng.standard_normal((10, 3))
    atom_fods = [elec_symbols[0]] * len(fods)
    if unrestricted:
        atom_fods += [elec_symbols[1]] * len(fods)
        fods = np.vstack((fods, rng.standard_normal((10, 3))))

    atom_split, pos_split, fods_split = split_fods(
        atom + atom_fods, np.vstack((pos, fods)), elec_symbols
    )
    assert_equal(atom, atom_split)
    assert_equal(pos, pos_split)
    if unrestricted:
        fods_split = np.vstack((fods_split[0], fods_split[1]))
    else:
        fods_split = fods_split[0]
    # Function is not stable, therefore sort arrays before the comparison
    fods = fods[fods[:, 0].argsort()]
    fods_split = fods_split[fods_split[:, 0].argsort()]
    assert_equal(fods, fods_split)


@pytest.mark.parametrize('unrestricted', [True, False])
def test_remove_core_fods(unrestricted):
    """Test core FOD removal function."""
    atoms = Atoms('Li5', rng.standard_normal((5, 3)), unrestricted=unrestricted).build()
    atoms.Z = 1
    core = atoms.pos
    valence = rng.standard_normal((10, 3))

    fods = [np.vstack((core, valence))] * atoms.occ.Nspin
    fods_valence = remove_core_fods(atoms, fods)
    assert_equal([valence] * atoms.occ.Nspin, fods_valence)


if __name__ == '__main__':
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
