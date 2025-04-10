# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]

def lda_c_chachiyo(
    n: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...
def chachiyo_scaling(
    zeta: _ArrayReal,
) -> tuple[_ArrayReal, _ArrayReal]: ...
def lda_c_chachiyo_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    weight_function: Callable[[_ArrayReal], tuple[_ArrayReal, _ArrayReal]] = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...
