# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def read_cube(
    filename: str,
) -> tuple[
    list[str],
    NDArray[np.float64],
    list[float],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]: ...
def write_cube(
    obj: Atoms | SCF,
    filename: str,
    field: NDArray[np.float64] | NDArray[np.complex128] | None,
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
