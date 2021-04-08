import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from numpy.random import seed, rand
from plainedft import Atoms
from plainedft import SCF
from plainedft.scf import getE, orth#, orth2
from plainedft.plot import *
from plainedft.tools import *
from plainedft.gth_nonloc import calc_Enl, calc_Vnl


def main(name):
    if name == 'Ne':
        atom = 'Ne'
        lattice = 16
        X = np.array([[8.000000003522448, 8.000000003522448, 8.000000003522448]])
        Z = 8
        Ns = 4
        S = 75 * np.array([1, 1, 1])
        f = 2
        ecut = 25
        verbose = 5
        pot = 'gth'

    if name == 'LiH':
        atom = ['Li', 'H']
        lattice = 16
        X = np.array([[0,0,0], [8.000000003522448, 8.000000003522448, 8.000000003522448]])
        Z = [1, 1]
        Ns = 1
        S = 75 * np.array([1, 1, 1])
        f = 2
        ecut = 25
        verbose = 5
        pot = 'gth'

    # Create atoms object, with pseudopotential parameters inside
    atoms = Atoms(atom=atom, a=lattice, X=X, Z=Z, Ns=Ns, S=S, f=f, ecut=ecut, verbose=verbose, pot=pot)

    # Create a basis set with a set seed
    seed(1234)
    W = rand(len(atoms.active[0]), atoms.Ns)
    # W = orth(atoms, W)

    E = calc_Enl(atoms, W)
    V = calc_Vnl(atoms, W)

    betaNL_psi = np.dot(W.T.conj(), atoms.betaNL).conj()

    return atoms, E, V, W, betaNL_psi

if __name__ == '__main__':
    from timeit import default_timer
    start = default_timer()
    print(f'Enl: {main("LiH")[1]}')
    end = default_timer()
    print(f'Time: {end-start}')
