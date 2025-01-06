# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from numpy import complexfloating, floating
from numpy.typing import NDArray

from ._typing import _Array2D, _IntArray

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
    a: NDArray[complexfloating],
    b: NDArray[complexfloating],
) -> float: ...
def Ylm_real(
    l: int,
    m: int,
    G: NDArray[floating],
) -> NDArray[floating]: ...
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
    size: _IntArray,
    seed: int = ...,
) -> NDArray[complexfloating]: ...
def add_maybe_none(
    a: NDArray[complexfloating] | None,
    b: NDArray[complexfloating] | None,
) -> NDArray[complexfloating]: ...
def molecule2list(molecule: str) -> list[str]: ...
def atom2charge(
    atom: Sequence[str],
    path: str | None = ...,
) -> list[int]: ...
def vector_angle(
    a: NDArray[floating],
    b: NDArray[floating],
) -> NDArray[floating]: ...
def get_lattice(lattice_vectors: _Array2D) -> list[NDArray[floating]]: ...
