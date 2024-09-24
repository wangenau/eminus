# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import overload, TypeVar

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

_AnyW = TypeVar("_AnyW", NDArray[complex128], list[NDArray[complex128]])

def eval_psi(
    atoms: Atoms,
    psi: _AnyW,
    r: Array1D | Array2D,
) -> _AnyW: ...
def get_R(
    atoms: Atoms,
    psi: _AnyW,
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> _AnyW: ...
def get_FO(
    atoms: Atoms,
    psi: _AnyW,
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> _AnyW: ...
def get_S(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[complex128]: ...
def get_FLO(
    atoms: Atoms,
    psi: _AnyW,
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> _AnyW: ...
def get_scdm(
    atoms: Atoms,
    psi: _AnyW,
) -> _AnyW: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
def wannier_supercell_matrices(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
def wannier_supercell_cost(
    X: NDArray[float64],
    Y: NDArray[float64],
    Z: NDArray[float64],
) -> float: ...
def wannier_supercell_grad(
    atoms: Atoms,
    X: NDArray[float64],
    Y: NDArray[float64],
    Z: NDArray[float64],
) -> NDArray[complex128]: ...
def get_wannier(
    atoms: Atoms,
    psirs: _AnyW,
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> _AnyW: ...
