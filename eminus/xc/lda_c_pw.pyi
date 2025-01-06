# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

from .._typing import _Array1D

def lda_c_pw(
    n: NDArray[floating],
    A: float = ...,
    a1: float = ...,
    b1: float = ...,
    b2: float = ...,
    b3: float = ...,
    b4: float = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
def lda_c_pw_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    A: _Array1D = ...,
    fzeta0: float = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
