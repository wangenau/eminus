# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import overload

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

@overload
def eval_psi(
    atoms: Atoms,
    psi: NDArray[complex128],
    r: Array1D | Array2D,
) -> NDArray[complex128]: ...
@overload
def eval_psi(
    atoms: Atoms,
    psi: list[NDArray[complex128]],
    r: Array1D | Array2D,
) -> list[NDArray[complex128]]: ...
@overload
def get_R(
    atoms: Atoms,
    psi: NDArray[complex128],
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> NDArray[complex128]: ...
@overload
def get_R(
    atoms: Atoms,
    psi: list[NDArray[complex128]],
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> list[NDArray[complex128]]: ...
@overload
def get_FO(
    atoms: Atoms,
    psi: NDArray[complex128],
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> NDArray[complex128]: ...
@overload
def get_FO(
    atoms: Atoms,
    psi: list[NDArray[complex128]],
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> list[NDArray[complex128]]: ...
def get_S(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[complex128]: ...
@overload
def get_FLO(
    atoms: Atoms,
    psi: NDArray[complex128],
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> NDArray[complex128]: ...
@overload
def get_FLO(
    atoms: Atoms,
    psi: list[NDArray[complex128]],
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> list[NDArray[complex128]]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
@overload
def wannier_cost(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
@overload
def wannier_center(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
@overload
def second_moment(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
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
@overload
def get_wannier(
    atoms: Atoms,
    psirs: NDArray[complex128],
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> NDArray[complex128]: ...
@overload
def get_wannier(
    atoms: Atoms,
    psirs: list[NDArray[complex128]],
    Nit: int = ...,
    conv_tol: float = ...,
    mu: float = ...,
    random_guess: bool = ...,
    seed: int | None = ...,
) -> list[NDArray[complex128]]: ...
