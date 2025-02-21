# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .occupations import Occupations
from .scf import SCF

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]
_ScalarOrArrayReal = TypeVar("_ScalarOrArrayReal", float, _ArrayReal)

def cutoff2gridspacing(E: float) -> float: ...
def gridspacing2cutoff(h: float) -> float: ...
def center_of_mass(
    coords: _ArrayReal,
    masses: Sequence[float] | _ArrayReal | None = ...,
) -> _ArrayReal: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: list[_ArrayComplex],
) -> list[_ArrayReal]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: _ArrayComplex,
) -> _ArrayReal: ...
def inertia_tensor(
    coords: _ArrayReal,
    masses: Sequence[float] | _ArrayReal | None = ...,
) -> _ArrayReal: ...
def get_dipole(
    scf: SCF,
    n: _ArrayReal | None = ...,
) -> _ArrayReal: ...
def get_ip(scf: SCF) -> float: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: list[_ArrayComplex],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: _ArrayComplex,
    eps: float = ...,
) -> bool: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: list[_ArrayComplex],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: _ArrayComplex,
    eps: float = ...,
) -> bool: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: list[_ArrayComplex],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: _ArrayComplex,
    eps: float = ...,
) -> bool: ...
def get_isovalue(
    n: _ArrayReal,
    percent: int = ...,
) -> float: ...
def get_tautf(scf: SCF) -> _ArrayReal: ...
def get_tauw(scf: SCF) -> _ArrayReal: ...
def get_elf(scf: SCF) -> _ArrayReal: ...
def get_reduced_gradient(
    scf: SCF,
    eps: float = ...,
) -> _ArrayReal: ...
def get_spin_squared(scf: SCF) -> float: ...
def get_multiplicity(scf: SCF) -> float: ...
def get_magnetization(scf: SCF) -> float: ...
def get_bandgap(scf: SCF) -> float: ...
def get_Efermi(
    obj: Occupations | SCF,
    epsilon: _ArrayReal | None = ...,
) -> float: ...
def fermi_distribution(
    E: _ScalarOrArrayReal,
    mu: float,
    kbT: float,
) -> _ScalarOrArrayReal: ...
def electronic_entropy(
    E: _ScalarOrArrayReal,
    mu: float,
    kbT: float,
) -> _ScalarOrArrayReal: ...
def get_dos(
    epsilon: _ArrayReal,
    wk: _ArrayReal,
    spin: int = ...,
    npts: int = ...,
    width: float = ...,
) -> tuple[_ArrayReal, _ArrayReal]: ...
