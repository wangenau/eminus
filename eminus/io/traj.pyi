# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

from eminus.atoms import Atoms
from eminus.scf import SCF

_Float: TypeAlias = floating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]

def read_traj(filename: str) -> list[tuple[list[str], _ArrayReal]]: ...
def write_traj(
    obj: Atoms | SCF,
    filename: str,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
