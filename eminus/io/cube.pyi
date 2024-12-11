# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def read_cube(
    filename: str,
) -> tuple[
    list[str],
    NDArray[floating],
    list[float],
    NDArray[floating],
    NDArray[integer],
    NDArray[floating],
]: ...
def write_cube(
    obj: Atoms | SCF,
    filename: str,
    field: NDArray[floating] | NDArray[complexfloating] | None,
    fods: NDArray[floating] | Sequence[NDArray[floating]] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
