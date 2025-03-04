# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the SCF class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, RSCF, SCF, USCF
from eminus.tools import center_of_mass

atoms = Atoms("He", (0, 0, 0), ecut=2, unrestricted=True)


def test_atoms():
    """Test that the Atoms object is independent."""
    scf = SCF(atoms)
    assert id(scf.atoms) != id(atoms)


def test_xc():
    """Test that xc functionals are correctly parsed."""
    scf = SCF(atoms, xc="LDA,VWN")
    assert scf.xc == ["lda_x", "lda_c_vwn"]
    assert scf.xc_type == "lda"
    assert scf.xc_params_defaults == {"A": 0.0310907, "b": 3.72744, "c": 12.9352, "x0": -0.10498}

    scf.xc = "PBE"
    assert scf.xc_type == "gga"
    assert scf.xc_params_defaults == {"beta": 0.06672455060314922, "mu": 0.2195149727645171}

    scf = SCF(atoms, xc=",")
    assert scf.xc == ["mock_xc", "mock_xc"]
    assert scf.xc_params_defaults == {}


def test_xc_params():
    """Test the custom xc parameters."""
    scf = SCF(atoms, xc="pbesol", opt={"sd": 1})
    scf.run()
    ref = scf.energies.Etot
    scf = SCF(atoms, xc="pbe", opt={"sd": 1})
    scf.xc_params = {
        "beta": 0.046,  # PBEsol parameter
        "mu": 10 / 81,  # PBEsol parameter
        "mock": 0,  # This should print a warning
    }
    scf.run()
    assert scf.energies.Etot == ref

    # Make sure default parameters are viable
    scf.xc_params = None
    scf.run()
    scf.xc_params = {}
    scf.run()


def test_pot():
    """Test that potentials are correctly parsed and initialized."""
    scf = SCF(atoms, pot="GTH")
    assert scf.pot == "gth"
    assert scf.psp == "pade"
    assert hasattr(scf, "gth")

    scf = SCF(atoms, pot="GTH", xc="pbe")
    assert scf.psp == "pbe"

    scf.pot = "test"
    assert scf.pot == "gth"
    assert scf.psp == "test"

    scf = SCF(atoms, pot="GE")
    assert scf.pot == "ge"
    assert not hasattr(scf, "gth")

    scf = SCF(atoms, pot="lr")
    assert scf.pot == "lr"
    assert scf.pot_params_defaults == {"alpha": 100}


def test_pot_params():
    """Test the custom pot parameters."""
    scf = SCF(atoms, pot="harmonic", opt={"sd": 1})
    scf.pot_params = {
        "freq": 1,
        "mock": 0,  # This should print a warning
    }
    scf.run()

    # Make sure default parameters are viable
    scf.pot_params = None
    scf.run()
    scf.pot_params = {}
    scf.run()


def test_guess():
    """Test initialization of the guess method."""
    scf = SCF(atoms, guess="RAND")
    assert scf.guess == "random"
    assert not scf.symmetric

    scf = SCF(atoms, guess="sym-pseudo")
    assert scf.guess == "pseudo"
    assert scf.symmetric


def test_gradtol():
    """Test the convergence depending of the gradient norm."""
    etot = SCF(atoms, etol=1, gradtol=1e-2).run()
    assert etot < -1


def test_sic():
    """Test that the SIC routine runs."""
    scf = SCF(atoms, xc="pbe", opt={"sd": 1}, sic=True)
    scf.run()
    assert scf.energies.Esic != 0


@pytest.mark.parametrize("disp", [True, {"atm": False}])
def test_disp(disp):
    """Test that the dispersion correction routine runs."""
    pytest.importorskip("dftd3", reason="dftd3 not installed, skip tests")
    scf = SCF(atoms, opt={"sd": 1}, disp=disp)
    scf.run()
    assert scf.energies.Edisp != 0


def test_symmetric():
    """Test the symmetry option for H2 dissociation."""
    atoms = Atoms("H2", ((0, 0, 0), (0, 0, 6)), ecut=1, unrestricted=True)
    scf_symm = SCF(atoms, guess="symm-rand")
    scf_unsymm = SCF(atoms, guess="unsymm-rand")
    assert scf_symm.run() > scf_unsymm.run()


def test_opt():
    """Test the optimizer option."""
    atoms = Atoms("He", (0, 0, 0), ecut=1)
    scf = SCF(atoms, opt={"AUTO": 1})
    assert "auto" in scf.opt
    scf.opt = {"sd": 1}
    assert "sd" in scf.opt
    assert "auto" not in scf.opt
    scf.run()
    assert hasattr(scf, "_opt_log")
    assert "sd" in scf._opt_log


def test_verbose():
    """Test the verbosity level."""
    scf = SCF(atoms)
    assert scf.verbose == atoms.verbose
    assert hasattr(atoms, "_log")
    assert hasattr(scf, "_log")
    assert scf._log.verbose == atoms._log.verbose

    level = "DEBUG"
    scf.verbose = level
    assert scf.verbose == level
    assert scf._log.verbose == level


def test_kpts():
    """Test the k-points object."""
    scf = SCF(atoms)
    assert id(scf.kpts) == id(scf.atoms.kpts)


def test_clear():
    """Test the clear function."""
    scf = SCF(atoms, opt={"sd": 1})
    scf.run()
    scf.clear()
    assert not scf.is_converged
    assert [
        x
        for x in (scf.Y, scf.n_spin, scf.dn_spin, scf.phi, scf.exc, scf.vxc, scf.vsigma, scf.vtau)
        if x is None
    ]


@pytest.mark.parametrize("center", [None, np.diag(atoms.a) / 2])
def test_recenter(center):
    """Test the recenter function."""
    scf = SCF(atoms)
    scf.run()
    Vloc = scf.Vloc
    assert scf.is_converged

    scf.recenter(center)
    assert scf.W is not None
    W = atoms.I(scf.W[0], 0)
    com = center_of_mass(scf.atoms.pos)
    # Check that the density is centered around the atom
    assert_allclose(center_of_mass(atoms.r, scf.n), com, atol=0.005)
    # Check that the orbitals are centered around the atom
    assert_allclose(
        center_of_mass(atoms.r, np.real(W[0, :, 0].conj() * W[0, :, 0])), com, atol=0.005
    )
    assert_allclose(
        center_of_mass(atoms.r, np.real(W[1, :, 0].conj() * W[1, :, 0])), com, atol=0.005
    )
    # Test that the local potential has been rebuild
    assert not np.array_equal(scf.Vloc, Vloc)
    assert scf.atoms.center == "recentered"


def test_rscf():
    """Test the RSCF object."""
    scf = RSCF(atoms)
    assert not scf.atoms.unrestricted
    assert atoms.occ.Nspin == 2
    assert id(scf.atoms) != id(atoms)


def test_uscf():
    """Test the USCF object."""
    scf = USCF(atoms)
    assert scf.atoms.unrestricted
    assert atoms.occ.Nspin == 2
    assert id(scf.atoms) != id(atoms)


def test_callback():
    """Test the callback method."""
    scf = SCF(atoms, opt={"pccg": 5})
    assert scf.callback(scf, 0) is None

    def callback(scf, step):  # noqa: ARG001
        """Force the wave function to ones so the SCF has to converge."""
        print("Get to the Chopper!")
        for ik in range(len(scf.W)):
            scf.W[ik][:] = 1

    scf.callback = callback
    scf.run()
    assert hasattr(scf, "_opt_log")
    assert scf._opt_log["pccg"]["iter"] == 2


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
