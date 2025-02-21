# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""PDB file handling."""

import numpy as np
from scipy.linalg import norm

from ..logger import log
from ..units import bohr2ang
from ..utils import vector_angle


def write_pdb(obj, filename, fods=None, elec_symbols=("X", "He"), trajectory=False):
    """Generate PDB files from atoms objects.

    See :func:`~eminus.io.pdb.create_pdb_str` for more information about the PDB file format.

    Args:
        obj: Atoms or SCF object.
        filename: PDB output file path/name.

    Keyword Args:
        fods: FOD coordinates to write.
        elec_symbols: Identifier for up and down FODs.
        trajectory: Allow appending to a file to create trajectories.
    """
    atoms = obj._atoms

    if not filename.endswith(".pdb"):
        filename += ".pdb"

    if "He" in atoms.atom and atoms.unrestricted:
        log.warning(
            'You need to modify "elec_symbols" to write helium with FODs in the spin-'
            "polarized case."
        )

    atom = atoms.atom
    pos = atoms.pos
    if fods is not None:
        if len(fods[0]) != 0:
            atom = atom + [elec_symbols[0]] * len(fods[0])
            pos = np.vstack((pos, fods[0]))
        if len(fods) > 1 and len(fods[1]) != 0:
            atom = atom + [elec_symbols[1]] * len(fods[1])
            pos = np.vstack((pos, fods[1]))

    # Append to a file when using the trajectory keyword
    if trajectory:
        mode = "a"
    else:
        mode = "w"

    with open(filename, mode, encoding="utf-8") as fp:
        fp.write(create_pdb_str(atom, pos, a=atoms.a))


def create_pdb_str(atom, pos, a=None):
    """Convert atom symbols and positions to the PDB format.

    File format definitions:
        CRYST1: https://wwpdb.org/documentation/file-format-content/format33/sect8.html#CRYST1

        ATOM: https://wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM

    Args:
        atom: Atom symbols.
        pos: Atom positions.

    Keyword Args:
        a: Cell size.

    Returns:
        PDB file format.
    """
    # Convert Bohr to Angstrom
    pos = bohr2ang(pos)
    if a is not None:
        a = bohr2ang(a)

    # PDB files have specific numbers of characters for every data with changing justification
    # Write everything explicitly down to not lose track of line lengths
    pdb = ""
    # Create data for a cuboidal cell
    if a is not None:
        pdb += "CRYST1"  # 1-6 "CRYST1"
        pdb += f"{norm(a[0]):>9,.3f}"  # 7-15 a
        pdb += f"{norm(a[1]):>9,.3f}"  # 16-24 b
        pdb += f"{norm(a[2]):>9,.3f}"  # 25-33 c
        pdb += f"{vector_angle(a[1], a[2]):>7,.2f}"  # 34-40 alpha
        pdb += f"{vector_angle(a[0], a[2]):>7,.2f}"  # 41-47 beta
        pdb += f"{vector_angle(a[0], a[1]):>7,.2f}"  # 48-54 gamma
        pdb += " "
        pdb += "P 1        "  # 56-66 Space group
        # pdb += "   1"                              # 67-70 Z value
        pdb += "\n"

    # Create molecule data
    pdb += "MODEL 1"
    for ia in range(len(atom)):
        pdb += "\nATOM  "  # 1-6 "ATOM"
        pdb += f"{ia + 1:>5}"  # 7-11 Atom serial number
        pdb += " "
        pdb += f"{atom[ia]:>4}"  # 13-16 Atom name
        pdb += " "  # 17 Alternate location indicator
        pdb += "MOL"  # 18-20 Residue name
        pdb += " "
        pdb += " "  # 22 Chain identifier
        pdb += "   1"  # 23-26 Residue sequence number
        pdb += " "  # 27 Code for insertions of residues
        pdb += "   "
        pdb += f"{pos[ia, 0]:>8,.3f}"  # 31-38 X orthogonal coordinate
        pdb += f"{pos[ia, 1]:>8,.3f}"  # 39-46 Y orthogonal coordinate
        pdb += f"{pos[ia, 2]:>8,.3f}"  # 47-54 Z orthogonal coordinate
        pdb += f"{1:>6,.2f}"  # 55-60 Occupancy
        pdb += f"{0:>6,.2f}"  # 61-66 Temperature factor
        pdb += "          "
        pdb += f"{atom[ia]:>2}"  # 77-78 Element symbol
        # pdb += "  "                  # 79-80 Charge
    return f"{pdb}\nENDMDL\n"
