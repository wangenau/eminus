# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]
_AnyW = TypeVar("_AnyW", _ArrayComplex, list[_ArrayComplex])

def get_phi(
    atoms: Atoms,
    n: _ArrayReal,
) -> _ArrayReal: ...
def orth(
    atoms: Atoms,
    W: _AnyW,
) -> _AnyW: ...
def orth_unocc(
    atoms: Atoms,
    Y: _AnyW,
    Z: _AnyW,
) -> _AnyW: ...
def get_n_total(
    atoms: Atoms,
    Y: list[_ArrayComplex],
    n_spin: _ArrayReal | None = ...,
) -> _ArrayReal: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: _ArrayComplex,
    ik: int,
) -> _ArrayReal: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: list[_ArrayComplex],
) -> _ArrayReal: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: _ArrayComplex,
    ik: int,
) -> _ArrayReal: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: list[_ArrayComplex],
) -> _ArrayReal: ...
def get_grad(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[_ArrayComplex],
    **kwargs: Any,
) -> _ArrayComplex: ...
def H(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[_ArrayComplex],
    dn_spin: _ArrayReal | None = ...,
    phi: _ArrayReal | None = ...,
    vxc: _ArrayComplex | None = ...,
    vsigma: _ArrayComplex | None = ...,
    vtau: _ArrayComplex | None = ...,
) -> _ArrayComplex: ...
def H_precompute(
    scf: SCF,
    W: list[_ArrayComplex],
) -> tuple[
    _ArrayReal,
    _ArrayComplex,
    _ArrayComplex,
    _ArrayComplex,
    _ArrayComplex,
]: ...
def Q(
    inp: _ArrayComplex,
    U: _ArrayComplex,
) -> _ArrayComplex: ...
def get_psi(
    scf: SCF,
    W: list[_ArrayComplex] | None,
    **kwargs: Any,
) -> list[_ArrayComplex]: ...
def get_epsilon(
    scf: SCF,
    W: list[_ArrayComplex] | None,
    **kwargs: Any,
) -> _ArrayReal: ...
def get_epsilon_unocc(
    scf: SCF,
    W: list[_ArrayComplex] | None,
    Z: list[_ArrayComplex] | None,
    **kwargs: Any,
) -> _ArrayReal: ...
def guess_random(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[_ArrayComplex]: ...
def guess_pseudo(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[_ArrayComplex]: ...
