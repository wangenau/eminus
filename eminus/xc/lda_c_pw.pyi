# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import floating
from numpy.typing import NDArray

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]

def lda_c_pw(
    n: _ArrayReal,
    A: float = ...,
    a1: float = ...,
    b1: float = ...,
    b2: float = ...,
    b3: float = ...,
    b4: float = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...
def lda_c_pw_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    A: Sequence[float] | _ArrayReal = ...,
    fzeta0: float = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...
