#!/usr/bin/env python3
'''Test energies for the implemented toy potentials.'''
from numpy.testing import assert_allclose

from eminus import Atoms, RSCF
from eminus.dft import get_epsilon
from eminus.units import ang2bohr


def test_harmonic():
    '''Compare total energies for the harmonic potential.'''
    atom = 'H'
    X = (0, 0, 0)
    a = 6
    ecut = 10
    s = (20, 25, 30)
    f = (2, 2, 2, 2)
    pot = 'harmonic'
    min = {'sd': 8, 'pccg': 51}

    atoms = Atoms(atom, X, a=a, ecut=ecut, s=s, f=f, Nspin=1)
    E = RSCF(atoms, pot=pot, guess='random', etol=1e-6, min=min).run()
    # We have to get close to the Etot reference value of 43.337 Eh (for different parameters)
    assert_allclose(E, 43.10344)


def test_coulomb():
    '''Compare total energies for the ionic potential.'''
    atom = 'H'
    X = (0, 0, 0)
    a = 10
    ecut = 10
    s = 20
    pot = 'coulomb'
    min = {'sd': 3, 'pccg': 11}

    atoms = Atoms(atom, X, a=a, ecut=ecut, s=s)
    E = RSCF(atoms, pot=pot, guess='random', etol=1e-6, min=min).run()
    # In the limit we should come close to the NIST Etot value of -0.445671 Eh
    assert_allclose(E, -0.438501)


def test_ge():
    '''Compare eigenstate energies for the germanium potential.'''
    atom = 'Ge'
    X = (0, 0, 0)
    a = ang2bohr(5.66)
    ecut = 10
    s = 10
    f = (2, 2 / 3, 2 / 3, 2 / 3)
    pot = 'ge'
    min = {'sd': 4, 'pccg': 23}

    atoms = Atoms(atom, X, a=a, ecut=ecut, s=s, Nspin=1, f=f)
    scf = RSCF(atoms, pot=pot, guess='random', etol=1e-6, min=min)
    scf.run()
    eps = get_epsilon(scf, scf.W)[0]
    # In the limit we should come close to the NIST 4s-4p value of -0.276641 Eh
    assert_allclose(eps[0] - eps[1], -0.2721017)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
