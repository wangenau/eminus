# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""POSCAR file handling."""

import time

import numpy as np

from ..logger import log
from ..units import ang2bohr, bohr2ang
from ..version import __version__


def read_poscar(filename):
    """Load atom species and positions from POSCAR files.

    File format definition: https://www.vasp.at/wiki/index.php/POSCAR

    Args:
        filename: POSCAR input file path/name.

    Returns:
        Atom species, positions, and cell vectors.
    """
    if "POSCAR" not in filename:
        filename += ".POSCAR"

    with open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()

        # The first line contains a comment, print it if available
        comment = lines[0].strip()
        if comment:
            log.info(f'POSCAR file comment: "{comment}"')

        # The second line contains scaling factors
        scaling = np.float64(lines[1].strip().split())

        # Followed by unscaled lattice coordinates, scale them here
        a = np.empty((3, 3))
        for i, line in enumerate(lines[2:5]):
            a[i] = scaling * np.float64(line.strip().split())

        # If line number six contains numbers a POTCAR file is required
        if lines[5].strip().split()[0].isnumeric():
            msg = "POTCAR files are not supported, provide species in the POSCAR file"
            raise NotImplementedError(msg)
        # Otherwise the atom species are given with their amount
        atom = lines[5].strip().split()
        Natom = np.int64(lines[6].strip().split())
        # Extend the atoms by their respective amount
        atom = [a for a, N in zip(atom, Natom) for _ in range(N)]

        # Ignore the dynamics line if available
        skip = 0
        if "dynamics" in lines[7]:
            skip += 1
        mode = lines[7 + skip].strip().lower()

        pos = np.empty((np.sum(Natom), 3))
        # Following lines contain atom positions
        for i, line in enumerate(lines[8 + skip : 8 + skip + np.sum(Natom)]):
            if mode == "direct":
                pos[i] = np.sum(a * np.float64(line.strip().split()[:3]), axis=0)
            if mode == "cartesian":
                pos[i] = scaling * np.float64(line.strip().split()[:3])
        # Skip all the properties afterwards

    # POSCAR files are in Angstrom, so convert to Bohr
    pos = ang2bohr(pos)
    a = ang2bohr(a)
    return atom, pos, a


def write_poscar(obj, filename, fods=None, elec_symbols=("X", "He")):
    """Generate POSCAR files from atoms objects.

    File format definition: https://www.vasp.at/wiki/index.php/POSCAR

    Args:
        obj: Atoms or SCF object.
        filename: POSCAR output file path/name.

    Keyword Args:
        fods: FOD coordinates to write.
        elec_symbols: Identifier for up and down FODs.
    """
    atoms = obj._atoms

    if "POSCAR" not in filename:
        filename += ".POSCAR"

    # Convert the coordinates from atomic units to Angstrom
    pos = bohr2ang(atoms.pos)
    a = bohr2ang(atoms.a)
    if fods is not None:
        fods = [bohr2ang(i) for i in fods]

    if "He" in atoms.atom and atoms.unrestricted:
        log.warning(
            'You need to modify "elec_symbols" to write helium with FODs in the spin-'
            "polarized case."
        )

    with open(filename, "w", encoding="utf-8") as fp:
        # Print information about the file and program, and the file creation time
        fp.write(f"File generated with eminus {__version__} on {time.ctime()}\n")

        # We have scaled vectors and coordinates
        fp.write("1.0\n")

        # Write lattice
        fp.write(
            f"{a[0, 0]:.6f} {a[0, 1]:.6f} {a[0, 2]:.6f}\n"
            f"{a[1, 0]:.6f} {a[1, 1]:.6f} {a[1, 2]:.6f}\n"
            f"{a[2, 0]:.6f} {a[2, 1]:.6f} {a[2, 2]:.6f}\n"
        )

        # We need to sort the atoms in the POSCAR file
        sort = np.argsort(atoms.atom)
        atom = np.asarray(atoms.atom)[sort]
        pos = pos[sort]

        # Write the sorted species
        fp.write(f"{' '.join(set(atom))}")
        if fods is not None:
            for s in range(len(fods)):
                if len(fods[s]) > 0:
                    fp.write(f" {elec_symbols[s]}")
        fp.write("\n")

        # Write the number per (sorted) species
        _, counts = np.unique(atom, return_counts=True)
        fp.write(f"{' '.join(map(str, counts))}")
        if fods is not None:
            for s in range(len(fods)):
                if len(fods[s]) > 0:
                    fp.write(f" {len(fods[s])}")
        fp.write("\nCartesian\n")

        # Write the coordinates
        fp.writelines(
            f"{pos[ia, 0]: .6f}  {pos[ia, 1]: .6f}  {pos[ia, 2]: .6f}\n"
            for ia in range(atoms.Natoms)
        )

        # Add FOD coordinates if needed
        if fods is not None:
            for s in range(len(fods)):
                fp.writelines(f"{ie[0]: .6f}  {ie[1]: .6f}  {ie[2]: .6f}\n" for ie in fods[s])
