# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import complexfloating
from numpy.typing import NDArray

_Complex: TypeAlias = complexfloating[Any]
_ArrayComplex: TypeAlias = NDArray[_Complex]

def is_array(value: Any) -> Any: ...
def expm(
    A: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> _ArrayComplex: ...
def sqrtm(
    A: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
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
