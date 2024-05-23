# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from .atoms import Atoms
from .typing import Array1D, Array2D, IntArray

STRUCTURES: dict[str, dict[str, str | list[list[float]]]]

def Cell(
    atom: str | Sequence[str],
    lattice: str | Array2D,
    ecut: float,
    a: float | Array2D | None,
    basis: Array1D | Array2D | None = ...,
    bands: int | None = ...,
    kmesh: int | IntArray = ...,
    smearing: float = ...,
    magnetization: float | None = ...,
    verbose: int | str | None = ...,
    **kwargs: Any,
) -> Atoms: ...
