# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from numpy import float64
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def write_pdb(
    obj: Atoms | SCF,
    filename: str,
    fods: NDArray[float64] | Sequence[NDArray[float64]] | None = ...,
    elec_symbols: Sequence[str] = ...,
    trajectory: bool = ...,
) -> None: ...
def create_pdb_str(
    atom: Sequence[str],
    pos: NDArray[float64],
    a: NDArray[float64] | None = ...,
) -> str: ...
