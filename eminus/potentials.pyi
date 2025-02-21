# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

from numpy import complexfloating
from numpy.typing import NDArray

from .scf import SCF

type _Complex = complexfloating[Any]
type _ArrayComplex = NDArray[_Complex]

def get_pot_defaults(pot: str) -> dict[str, float]: ...
def harmonic(
    scf: SCF,
    freq: float = ...,
    **kwargs: Any,
) -> _ArrayComplex: ...
def coulomb(
    scf: SCF,
    **kwargs: Any,
) -> _ArrayComplex: ...
def coulomb_lr(
    scf: SCF,
    alpha: float = ...,
    **kwargs: Any,
) -> _ArrayComplex: ...
def ge(
    scf: SCF,
    **kwargs: Any,
) -> _ArrayComplex: ...
def init_pot(
    scf: SCF,
    pot_params: dict[str, float] | None = ...,
) -> _ArrayComplex: ...

IMPLEMENTED: dict[str, Callable[[SCF], _ArrayComplex]]
