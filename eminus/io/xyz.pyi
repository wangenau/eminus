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

def read_xyz(filename: str) -> tuple[list[str], _ArrayReal]: ...
def write_xyz(
    obj: Atoms | SCF,
    filename: str,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    elec_symbols: Sequence[str] = ...,
    trajectory: bool = ...,
) -> None: ...
