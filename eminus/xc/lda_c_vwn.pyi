# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

def lda_c_vwn(
    n: NDArray[floating],
    A: float = ...,
    b: float = ...,
    c: float = ...,
    x0: float = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
def lda_c_vwn_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
