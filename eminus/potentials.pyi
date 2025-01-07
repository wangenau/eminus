# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

from numpy import complexfloating
from numpy.typing import NDArray

from .scf import SCF

type _Complex = complexfloating[Any]
type _ArrayComplex = NDArray[_Complex]

def harmonic(scf: SCF) -> _ArrayComplex: ...
def coulomb(scf: SCF) -> _ArrayComplex: ...
def ge(scf: SCF) -> _ArrayComplex: ...
def init_pot(scf: SCF) -> _ArrayComplex: ...

IMPLEMENTED: dict[str, Callable[[SCF], _ArrayComplex]]
