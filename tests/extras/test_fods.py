#!/usr/bin/env python3
'''Test fods identities.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_equal
import pytest

from eminus import Atoms, SCF
from eminus.dft import get_psi
from eminus.extras import pycom, remove_core_fods, split_fods

rng = default_rng()


@pytest.mark.extras
@pytest.mark.parametrize('Nspin', [1, 2])
@pytest.mark.parametrize('basis, loc, elec_symbols', [('pc-0', 'fb', ['X', 'He']),
                                                      ('pc-1', 'fb', ['X', 'He']),
                                                      ('pc-0', 'er', ['X', 'He']),
                                                      ('pc-0', 'fb', ['He', 'Ne'])])
def test_get_fods_unpol(Nspin, basis, loc, elec_symbols):
    '''Test FOD generator.'''
    from eminus.extras import get_fods
    atoms = Atoms('He', (0, 0, 0), Nspin=Nspin).build()
    fods = get_fods(atoms, basis=basis, loc=loc, elec_symbols=elec_symbols)
    # For He all FODs are core FODs and therefore should be close to the atom
    assert_allclose(atoms.X, fods[0], atol=1e-6)


@pytest.mark.parametrize('Nspin', [1, 2])
@pytest.mark.parametrize('elec_symbols', (['X', 'He'], ['He', 'Ne']))
def test_split_fods(Nspin, elec_symbols):
    '''Test splitting FODs from atoms.'''
    X = rng.standard_normal((5, 3))
    atom = ['H'] * len(X)
    fods = rng.standard_normal((10, 3))
    atom_fods = [elec_symbols[0]] * len(fods)
    if Nspin == 2:
        atom_fods += [elec_symbols[1]] * len(fods)
        fods = np.vstack((fods, rng.standard_normal((10, 3))))

    atom_split, X_split, fods_split = split_fods(atom + atom_fods, np.vstack((X, fods)),
                                                 elec_symbols)
    assert_equal(atom, atom_split)
    assert_equal(X, X_split)
    if Nspin == 2:
        fods_split = np.vstack((fods_split[0], fods_split[1]))
    else:
        fods_split = fods_split[0]
    # Function is not stable, therefore sort arrays before the comparison
    fods = fods[fods[:, 0].argsort()]
    fods_split = fods_split[fods_split[:, 0].argsort()]
    assert_equal(fods, fods_split)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_remove_core_fods(Nspin):
    '''Test core FOD removal function.'''
    atoms = Atoms('Li5', rng.standard_normal((5, 3)), Nspin=Nspin).build()
    core = atoms.X
    valence = rng.standard_normal((10, 3))

    fods = [np.vstack((core, valence))] * Nspin
    fods_valence = remove_core_fods(atoms, fods)
    assert_equal([valence] * Nspin, fods_valence)


@pytest.mark.parametrize('Nspin', [1, 2])
def test_pycom(Nspin):
    '''Test PyCOM routine.'''
    atoms = Atoms('He2', ((0, 0, 0), (10, 0, 0)), s=10, Nspin=Nspin, center=True).build()
    scf = SCF(atoms)
    scf.run()
    psi = atoms.I(get_psi(scf, scf.W))
    for spin in range(Nspin):
        assert_allclose(pycom(atoms, psi)[spin], [[10] * 3] * 2, atol=1e-1)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
