# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from numpy import complex128, float64
from numpy.typing import NDArray

from .typing import Array2D, IntArray

# Create a custom Callable type for some decorators
class _HandleType(Protocol):
    def __call__(
        self,
        obj: Any,
        W: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

class BaseObject:
    def view(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    def write(
        self,
        filename: str,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

def dotprod(
    a: NDArray[complex128],
    b: NDArray[complex128],
) -> float: ...
def Ylm_real(
    l: int,
    m: int,
    G: NDArray[float64],
) -> NDArray[float64]: ...
def handle_spin(
    func: _HandleType,
) -> _HandleType: ...
def handle_k(
    func: _HandleType | None = ...,
    *,
    mode: Literal["gracefully", "index", "reduce", "skip"] = ...,
) -> Any: ...
def handle_torch(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
def pseudo_uniform(
    size: IntArray,
    seed: int = ...,
) -> NDArray[complex128]: ...
def add_maybe_none(
    a: NDArray[complex128] | None,
    b: NDArray[complex128] | None,
) -> NDArray[complex128]: ...
def molecule2list(molecule: str) -> list[str]: ...
def atom2charge(
    atom: Sequence[str],
    path: str | None = ...,
) -> list[int]: ...
def vector_angle(
    a: NDArray[float64],
    b: NDArray[float64],
) -> NDArray[float64]: ...
def get_lattice(lattice_vectors: Array2D) -> list[NDArray[float64]]: ...
