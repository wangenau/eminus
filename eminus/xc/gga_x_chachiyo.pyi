# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

def gga_x_chachiyo(
    n: NDArray[floating],
    dn_spin: NDArray[floating] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
def gga_x_chachiyo_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    dn_spin: NDArray[floating] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
