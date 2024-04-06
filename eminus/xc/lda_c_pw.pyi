# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

def lda_c_pw(
    n: NDArray[np.float64],
    A: float = ...,
    a1: float = ...,
    b1: float = ...,
    b2: float = ...,
    b3: float = ...,
    b4: float = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
def lda_c_pw_spin(
    n: NDArray[np.float64],
    zeta: NDArray[np.float64],
    A: Sequence[float] = ...,
    fzeta0: float = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...