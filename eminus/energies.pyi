# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, overload

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]

@dataclass
class Energy:
    Ekin: float = ...
    Ecoul: float = ...
    Exc: float = ...
    Eloc: float = ...
    Enonloc: float = ...
    Eewald: float = ...
    Esic: float = ...
    Edisp: float = ...
    Eentropy: float = ...
    @property
    def Etot(self) -> float: ...
    def extrapolate(self) -> float: ...

def get_E(scf: SCF) -> float: ...
@overload
def get_Ekin(
    atoms: Atoms,
    Y: _ArrayComplex,
    ik: int,
) -> float: ...
@overload
def get_Ekin(
    atoms: Atoms,
    Y: list[_ArrayComplex],
) -> float: ...
def get_Ecoul(
    atoms: Atoms,
    n: _ArrayReal,
    phi: _ArrayReal | None = ...,
) -> float: ...
def get_Exc(
    scf: SCF,
    n: _ArrayReal,
    exc: _ArrayReal | None = ...,
    n_spin: _ArrayReal | None = ...,
    dn_spin: _ArrayReal | None = ...,
    tau: _ArrayReal | None = ...,
    Nspin: int = ...,
) -> float: ...
def get_Eloc(
    scf: SCF,
    n: _ArrayReal,
) -> float: ...
@overload
def get_Enonloc(
    scf: SCF,
    Y: _ArrayComplex,
    ik: int,
) -> float: ...
@overload
def get_Enonloc(
    scf: SCF,
    Y: list[_ArrayComplex],
) -> float: ...
def get_Eewald(
    atoms: Atoms,
    gcut: float = ...,
    gamma: float = ...,
) -> float: ...
def get_Esic(
    scf: SCF,
    Y: list[_ArrayComplex] | None,
    n_single: _ArrayReal | None = ...,
) -> float: ...
def get_Edisp(
    scf: SCF,
    version: str = ...,
    atm: bool = ...,
    xc: str | None = ...,
) -> float: ...
def get_Eband(
    scf: SCF,
    Y: list[_ArrayComplex],
    **kwargs: Any,
) -> float: ...
def get_Eentropy(
    scf: SCF,
    epsilon: _ArrayReal,
    Efermi: float,
) -> float: ...
