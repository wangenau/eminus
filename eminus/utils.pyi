# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias

from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

_Int: TypeAlias = integer[Any]
_Float: TypeAlias = floating[Any]
_Complex: TypeAlias = complexfloating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayComplex: TypeAlias = NDArray[_Complex]
_Array2D: TypeAlias = Sequence[Sequence[float]] | Sequence[_ArrayReal] | _ArrayReal

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
def sqrtm(A: _ArrayComplex) -> _ArrayComplex: ...
def handle_spin(
    func: _HandleType,
) -> _HandleType: ...
def handle_k(
    func: _HandleType | None = ...,
    *,
    mode: Literal["gracefully", "index", "reduce", "skip"] = ...,
) -> Any: ...
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
