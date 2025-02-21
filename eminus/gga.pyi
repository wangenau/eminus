# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]

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
