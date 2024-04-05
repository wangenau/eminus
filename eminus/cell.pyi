# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Sequence

from .atoms import Atoms
from .typing import Array2D, IntArray

STRUCTURES: dict[str, dict[str, str | list[list[float]]]]

def Cell(
    atom: str | Sequence[str],
    lattice: str | Array2D,
    ecut: float,
    a: float | Array2D | None,
    basis: Array2D | None = ...,
    bands: int | None = ...,
    kmesh: int | IntArray = ...,
    smearing: float = ...,
    verbose: int | str | None = ...,
    **kwargs: Any,
) -> Atoms: ...
