# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def get_localized_orbitals(
    mf: Any,
    Nspin: int,
    loc: str,
    Nit: int = ...,
    seed: int = ...,
) -> list[NDArray[np.float64]]: ...
def get_fods(
    obj: Atoms | SCF,
    basis: str = ...,
    loc: str = ...,
) -> list[NDArray[np.float64]]: ...
def split_fods(
    atom: Sequence[str],
    pos: NDArray[np.float64],
    elec_symbols: Sequence[str] = ...,
) -> tuple[list[str], NDArray[np.float64], list[NDArray[np.float64]]]: ...
def remove_core_fods(
    obj: Atoms | SCF,
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.float64]]: ...
