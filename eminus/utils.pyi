# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

type _Int = integer[Any]
type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]
type _Array2D = Sequence[Sequence[float]] | Sequence[_ArrayReal] | _ArrayReal

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
    a: _ArrayComplex,
    b: _ArrayComplex,
) -> float: ...
def Ylm_real(
    l: int,
    m: int,
    G: _ArrayReal,
) -> _ArrayReal: ...
def handle_spin(
    func: _HandleType,
) -> _HandleType: ...
def handle_k(
    func: _HandleType | None = ...,
    *,
    mode: Literal["gracefully", "index", "reduce", "skip"] = ...,
) -> Any: ...
def handle_backend(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
def pseudo_uniform(
    size: Sequence[int] | NDArray[_Int],
    seed: int = ...,
) -> _ArrayComplex: ...
def add_maybe_none(
    a: _ArrayComplex | None,
    b: _ArrayComplex | None,
) -> _ArrayComplex: ...
def molecule2list(molecule: str) -> list[str]: ...
def atom2charge(
    atom: Sequence[str],
    path: str | None = ...,
) -> list[int]: ...
def vector_angle(
    a: _ArrayReal,
    b: _ArrayReal,
) -> _ArrayReal: ...
def get_lattice(lattice_vectors: _Array2D) -> list[_ArrayReal]: ...
