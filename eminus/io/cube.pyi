# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import TypeAlias

from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

from eminus.atoms import Atoms
from eminus.scf import SCF

_Int: TypeAlias = integer
_Float: TypeAlias = floating
_Complex: TypeAlias = complexfloating
_ArrayReal: TypeAlias = NDArray[_Float]

def read_cube(
    filename: str,
) -> tuple[
    list[str],
    _ArrayReal,
    list[float],
    _ArrayReal,
    NDArray[_Int],
    _ArrayReal,
]: ...
def write_cube(
    obj: Atoms | SCF,
    filename: str,
    field: NDArray[_Float | _Complex] | None,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
