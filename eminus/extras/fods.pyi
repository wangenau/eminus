# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import floating
from numpy.typing import NDArray

from ..atoms import Atoms
from ..scf import SCF

def get_localized_orbitals(
    mf: Any,
    Nspin: int,
    loc: str,
    Nit: int = ...,
    seed: int = ...,
) -> list[NDArray[floating]]: ...
def get_fods(
    obj: Atoms | SCF,
    basis: str = ...,
    loc: str = ...,
) -> list[NDArray[floating]]: ...
def split_fods(
    atom: Sequence[str],
    pos: NDArray[floating],
    elec_symbols: Sequence[str] = ...,
) -> tuple[list[str], NDArray[floating], list[NDArray[floating]]]: ...
def remove_core_fods(
    obj: Atoms | SCF,
    fods: NDArray[floating] | Sequence[NDArray[floating]],
) -> list[NDArray[floating]]: ...
