# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test HDF5 input and output functionalities."""

import os

import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, Cell, SCF
from eminus.energies import Energy
from eminus.extras import read_hdf5, write_hdf5
from eminus.gth import GTH
from eminus.io import read, write
from eminus.kpoints import KPoints
from eminus.occupations import Occupations

atoms = Atoms("LiH", ((0, 0, 0), (3, 0, 0)), ecut=1).build()
scf = SCF(atoms, opt={"sd": 1})
scf.run()


@pytest.mark.parametrize(
    "obj",
    [
        atoms,
        atoms.kpts,
        atoms.occ,
        scf,
        scf.energies,
        scf.gth,
        Atoms("He", (0, 0, 0)),
        KPoints("sc"),
        Occupations(),
        SCF(atoms),
        Energy(),
        GTH(),
    ],
)
def test_hdf5(obj):
    """Test HDF5 file output and input."""
    pytest.importorskip("h5py", reason="h5py not installed, skip tests")
    filename = "test.hdf5"
    write_hdf5(obj, filename)
    test = read(filename)
    os.remove(filename)
    for attr in test.__dict__:
        # Skip objects and dictionaries
        if attr in {"_atoms", "gth", "kpts", "_log", "_precomputed"}:
            continue
        if attr == "GTH":
            assert getattr(obj, attr).keys() == getattr(test, attr).keys()
            continue
        try:
            assert_allclose(getattr(obj, attr), getattr(test, attr))
        except TypeError:
            assert getattr(obj, attr) == getattr(test, attr)
        except ValueError:
            for i in range(len(getattr(obj, attr))):
                assert_allclose(getattr(obj, attr)[i], getattr(test, attr)[i])


def test_hdf5_restart():
    """Test the SCF restart from HDF5 files."""
    pytest.importorskip("h5py", reason="h5py not installed, skip tests")
    filename = "test.hdf5"
    write(scf, filename)
    test_scf = read(filename)
    test_scf.run()
    os.remove(filename)

    cell = Cell("He", "sc", 1, 1)
    scf_cell = SCF(cell, opt={"sd": 1})
    scf_cell.run()
    write(scf_cell, filename)
    test_scf = read(filename)
    test_scf.run()
    os.remove(filename)


def test_filename_ending():
    """Test if the HDF5 functions still work when omitting the filename ending."""
    pytest.importorskip("h5py", reason="h5py not installed, skip tests")
    filename = "test"
    write_hdf5(atoms, filename)
    read_hdf5(filename)
    os.remove(f"{filename}.hdf5")


def test_write_method():
    """Test the HDF5 file writing using the write method."""
    pytest.importorskip("h5py", reason="h5py not installed, skip tests")
    filename = "test"
    scf.write(filename + ".hdf5")
    os.remove(f"{filename}.hdf5")


@pytest.mark.parametrize(
    ("compression", "compression_opts"), [("gzip", 0), ("gzip", 9), ("lzf", None)]
)
def test_compression(compression, compression_opts):
    """Test the HDF5 compression filters."""
    pytest.importorskip("h5py", reason="h5py not installed, skip tests")
    filename = "test"
    write_hdf5(atoms, filename, compression=compression, compression_opts=compression_opts)
    read_hdf5(filename)
    os.remove(f"{filename}.hdf5")


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
