# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import complex128, float64, int64
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def read_cube(
    filename: str,
) -> tuple[
    list[str],
    NDArray[float64],
    list[float],
    NDArray[float64],
    NDArray[int64],
    NDArray[float64],
]: ...
def write_cube(
    obj: Atoms | SCF,
    filename: str,
    field: NDArray[float64] | NDArray[complex128] | None,
    fods: NDArray[float64] | Sequence[NDArray[float64]] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
