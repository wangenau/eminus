# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from ._typing import _Array1D, _Array2D, _IntArray
from .atoms import Atoms

STRUCTURES: dict[str, dict[str, str | list[list[float]]]]

def Cell(
    atom: str | Sequence[str],
    lattice: str | _Array2D,
    ecut: float,
    a: float | _Array2D | None,
    basis: _Array1D | _Array2D | None = ...,
    bands: int | None = ...,
    kmesh: int | _IntArray = ...,
    smearing: float = ...,
    magnetization: float | None = ...,
    verbose: int | str | None = ...,
    **kwargs: Any,
) -> Atoms: ...
