# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

_AnyW = TypeVar("_AnyW", NDArray[complexfloating], list[NDArray[complexfloating]])

def eval_psi(
    atoms: Atoms,
    psi: _AnyW,
    r: Array1D | Array2D,
) -> _AnyW: ...
def get_R(
    atoms: Atoms,
    psi: _AnyW,
    fods: NDArray[floating] | Sequence[NDArray[floating]],
) -> _AnyW: ...
def get_FO(
    atoms: Atoms,
    psi: _AnyW,
    fods: NDArray[floating] | Sequence[NDArray[floating]],
) -> _AnyW: ...
def get_S(
    atoms: Atoms,
    psirs: NDArray[complexfloating],
) -> NDArray[complexfloating]: ...
def get_FLO(
    atoms: Atoms,
    psi: _AnyW,
    fods: NDArray[floating] | Sequence[NDArray[floating]],
) -> _AnyW: ...
def get_scdm(
    atoms: Atoms,
    psi: _AnyW,
) -> _AnyW: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: list[NDArray[complexfloating]],
) -> list[NDArray[floating]]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: NDArray[complexfloating],
) -> NDArray[floating]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: list[NDArray[complexfloating]],
) -> list[NDArray[floating]]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: NDArray[complexfloating],
) -> NDArray[floating]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: list[NDArray[complexfloating]],
) -> list[NDArray[floating]]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: NDArray[complexfloating],
) -> NDArray[floating]: ...
def wannier_supercell_matrices(
    atoms: Atoms,
    psirs: NDArray[complexfloating],
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
def wannier_supercell_cost(
    X: NDArray[floating],
    Y: NDArray[floating],
    Z: NDArray[floating],
) -> float: ...
def wannier_supercell_grad(
    atoms: Atoms,
    X: NDArray[floating],
    Y: NDArray[floating],
    Z: NDArray[floating],
) -> NDArray[complexfloating]: ...
def get_wannier(
    atoms: Atoms,
    psirs: _AnyW,
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> _AnyW: ...
