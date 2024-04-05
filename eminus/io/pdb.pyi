# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def write_pdb(
    obj: Atoms | SCF,
    filename: str,
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]] | None = ...,
    elec_symbols: Sequence[str] = ...,
    trajectory: bool = ...,
) -> None: ...
def create_pdb_str(
    atom: Sequence[str],
    pos: NDArray[np.float64],
    a: NDArray[np.float64] | None = ...,
) -> str: ...
