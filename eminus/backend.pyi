# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, overload, TypeAlias

from numpy import complexfloating, integer
from numpy.typing import NDArray

_Int: TypeAlias = integer[Any]
_Complex: TypeAlias = complexfloating[Any]
_ArrayInt: TypeAlias = NDArray[_Int]
_ArrayComplex: TypeAlias = NDArray[_Complex]

def __getattr__(name: str) -> Any: ...
def is_array(value: _ArrayComplex) -> bool: ...
@overload
def to_np(*args: _ArrayComplex) -> _ArrayComplex: ...
@overload
def to_np(*args: tuple[_ArrayComplex]) -> tuple[_ArrayComplex]: ...
def delete(
    arr: _ArrayComplex,
    obj: int | Sequence[int] | _ArrayInt,
    axis: int | None = ...,
) -> _ArrayComplex: ...
def fftn(
    x: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> _ArrayComplex: ...
def ifftn(
    x: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> _ArrayComplex: ...
def sqrtm(
    A: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> _ArrayComplex: ...
