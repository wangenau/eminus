# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import floating
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]

def write_pdb(
    obj: Atoms | SCF,
    filename: str,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    elec_symbols: Sequence[str] = ...,
    trajectory: bool = ...,
) -> None: ...
def create_pdb_str(
    atom: Sequence[str],
    pos: _ArrayReal,
    a: _ArrayReal | None = ...,
) -> str: ...
