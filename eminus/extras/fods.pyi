# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import float64
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def get_localized_orbitals(
    mf: Any,
    Nspin: int,
    loc: str,
    Nit: int = ...,
    seed: int = ...,
) -> list[NDArray[float64]]: ...
def get_fods(
    obj: Atoms | SCF,
    basis: str = ...,
    loc: str = ...,
) -> list[NDArray[float64]]: ...
def split_fods(
    atom: Sequence[str],
    pos: NDArray[float64],
    elec_symbols: Sequence[str] = ...,
) -> tuple[list[str], NDArray[float64], list[NDArray[float64]]]: ...
def remove_core_fods(
    obj: Atoms | SCF,
    fods: NDArray[float64] | Sequence[NDArray[float64]],
) -> list[NDArray[float64]]: ...
