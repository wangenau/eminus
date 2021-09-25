from numpy.testing import assert_allclose

from plainedft import Atoms, SCF, read_xyz, __path__


def test_spin_paired():
    '''Compare total energies for small test-systems with reference values.'''
    # Total energies calculated with PWDFT.jl for H, H2, LiH, CH4, and Ne with same parameters
    # These values can be generated with the file ref_spin_paired.jl
    Etot_ref = [-0.4353327, -1.10228799, -0.76526088, -7.71058153, -29.88936935]

    path = f'{__path__[0]}/../tests/spin_paired'
    systems = ['H', 'H2', 'LiH', 'CH4', 'Ne']
    a = 16
    ecut = 10
    S = 48

    Etot = []
    for sys in systems:
        atom, X = read_xyz(f'{path}/{sys}.xyz')
        atoms = Atoms(atom=atom, X=X, a=a, ecut=ecut, S=S)
        e = SCF(atoms)
        Etot.append(e)

    assert_allclose(Etot, Etot_ref)
    return


if __name__ == '__main__':
    test_spin_paired()
