from numpy.testing import assert_allclose

from plainedft import Atoms, SCF, read_xyz, __path__


def test_spin_paired():
    # Total energies calculated with PWDFT.jl for H, H2, LiH, CH4, and Ne with same parameters
    Etot_ref = [-0.4353327, -1.10228799, -6.48192716, -7.71058154, -29.88936935]

    path = f'{__path__[0]}/../tests/spin_paired'
    systems = ['H', 'H2', 'LiH', 'CH4', 'Ne']
    a = 16
    ecut = 10
    S = 48

    for i, sys in enumerate(systems):
        atom, X = read_xyz(f'{path}/{sys}.xyz')
        atoms = Atoms(atom=atom, X=X, a=a, ecut=ecut, S=S)
        Etot = SCF(atoms)
        assert_allclose(Etot, Etot_ref[i])


if __name__ == '__main__':
    test_spin_paired()
