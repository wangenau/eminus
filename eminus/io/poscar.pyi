# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import floating
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def read_poscar(filename: str) -> tuple[list[str], NDArray[floating]]: ...
def write_poscar(
    obj: Atoms | SCF,
    filename: str,
    fods: NDArray[floating] | Sequence[NDArray[floating]] | None = ...,
    elec_symbols: Sequence[str] = ...,
) -> None: ...
