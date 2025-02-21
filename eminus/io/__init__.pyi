# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .cube import read_cube, write_cube
from .gth import read_gth
from .json import read_json, write_json
from .pdb import create_pdb_str, write_pdb
from .poscar import read_poscar, write_poscar
from .traj import read_traj, write_traj
from .xyz import read_xyz, write_xyz

__all__: list[str] = [
    "create_pdb_str",
    "read",
    "read_cube",
    "read_gth",
    "read_json",
    "read_poscar",
    "read_traj",
    "read_xyz",
    "write",
    "write_cube",
    "write_json",
    "write_pdb",
    "write_poscar",
    "write_traj",
    "write_xyz",
]

def read(
    filename: str,
    *args: Any,
    **kwargs: Any,
) -> Any: ...
def write(
    obj: Any,
    filename: str,
    *args: Any,
    **kwargs: Any,
) -> None: ...
