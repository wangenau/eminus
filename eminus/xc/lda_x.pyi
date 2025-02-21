# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]

def lda_x(
    n: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...
def lda_x_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...
