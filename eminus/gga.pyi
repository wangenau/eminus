# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload, TypeAlias

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

_Float: TypeAlias = floating[Any]
_Complex: TypeAlias = complexfloating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayComplex: TypeAlias = NDArray[_Complex]

def get_grad_field(
    atoms: Atoms,
    field: _ArrayReal | _ArrayComplex,
    real: bool = ...,
) -> _ArrayReal | _ArrayComplex: ...
def gradient_correction(
    atoms: Atoms,
    spin: int,
    dn_spin: _ArrayReal,
    vsigma: _ArrayComplex,
) -> _ArrayComplex: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: _ArrayComplex,
    ik: int,
) -> _ArrayReal: ...
@overload
def get_tau(
    atoms: Atoms,
    Y: list[_ArrayComplex],
) -> _ArrayReal: ...
def calc_Vtau(
    scf: SCF,
    ik: int,
    spin: int,
    W: _ArrayComplex,
    vtau: _ArrayComplex,
) -> _ArrayComplex: ...
