# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import overload, TypeAlias, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .utils import handle_k, handle_spin

_Float: TypeAlias = floating
_Complex: TypeAlias = complexfloating
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayComplex: TypeAlias = NDArray[_Complex]
_Array1D: TypeAlias = Sequence[float] | _ArrayReal
_Array2D: TypeAlias = Sequence[_Array1D] | _ArrayReal
_AnyW = TypeVar("_AnyW", _ArrayComplex, list[_ArrayComplex])

@handle_k(mode="skip")
def eval_psi(
    atoms: Atoms,
    psi: _AnyW,
    r: _Array1D | _Array2D,
) -> _AnyW: ...
@handle_k(mode="skip")
def get_R(
    atoms: Atoms,
    psi: _AnyW,
    fods: _ArrayReal | Sequence[_ArrayReal],
) -> _AnyW: ...
@handle_k(mode="skip")
def get_FO(
    atoms: Atoms,
    psi: _AnyW,
    fods: _ArrayReal | Sequence[_ArrayReal],
) -> _AnyW: ...
@handle_spin
def get_S(
    atoms: Atoms,
    psirs: _ArrayComplex,
) -> _ArrayComplex: ...
@handle_k(mode="skip")
def get_FLO(
    atoms: Atoms,
    psi: _AnyW,
    fods: _ArrayReal | Sequence[_ArrayReal],
) -> _AnyW: ...
@handle_k(mode="skip")
@handle_spin
def get_scdm(
    atoms: Atoms,
    psi: _AnyW,
) -> _AnyW: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: list[_ArrayComplex],
) -> list[_ArrayReal]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: _ArrayComplex,
) -> _ArrayReal: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: list[_ArrayComplex],
) -> list[_ArrayReal]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: _ArrayComplex,
) -> _ArrayReal: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: list[_ArrayComplex],
) -> list[_ArrayReal]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: _ArrayComplex,
) -> _ArrayReal: ...
@handle_spin
def wannier_supercell_matrices(
    atoms: Atoms,
    psirs: _ArrayComplex,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def wannier_supercell_cost(
    X: _ArrayReal,
    Y: _ArrayReal,
    Z: _ArrayReal,
) -> float: ...
def wannier_supercell_grad(
    atoms: Atoms,
    X: _ArrayReal,
    Y: _ArrayReal,
    Z: _ArrayReal,
) -> _ArrayComplex: ...
@handle_k(mode="skip")
@handle_spin
def get_wannier(
    atoms: Atoms,
    psirs: _AnyW,
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> _AnyW: ...
