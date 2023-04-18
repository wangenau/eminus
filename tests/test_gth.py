#!/usr/bin/env python3
'''Test GTH functions.'''
import numpy as np
from numpy.testing import assert_allclose

from eminus import Atoms, SCF


def test_norm():
    '''Test the norm of the GTH projector functions.'''
    # Actinium is the perfect test case since it tests all cases in eval_proj_G with the minimum
    # number of electrons (q-11)
    # The norm depends on the cut-off energy but will converge to 1
    # Choose an ecut such that we get at least 0.9
    atoms = Atoms('Ac', (0, 0, 0), ecut=35)
    scf = SCF(atoms)
    for i in range(scf.NbetaNL):
        norm = np.sum(scf.betaNL[:, i] * scf.betaNL[:, i])
        assert_allclose(abs(norm), 1, atol=1e-1)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
