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

def get_localized_orbitals(
    mf: Any,
    loc: str,
    Nit: int = ...,
    seed: int = ...,
) -> list[_ArrayReal]: ...
def get_fods(
    obj: Atoms | SCF,
    basis: str = ...,
    loc: str = ...,
) -> list[_ArrayReal]: ...
def split_fods(
    atom: Sequence[str],
    pos: _ArrayReal,
    elec_symbols: Sequence[str] = ...,
) -> tuple[list[str], _ArrayReal, list[_ArrayReal]]: ...
def remove_core_fods(
    obj: Atoms | SCF,
    fods: _ArrayReal | Sequence[_ArrayReal],
) -> list[_ArrayReal]: ...
