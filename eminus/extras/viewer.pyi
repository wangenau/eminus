# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import floating
from numpy.typing import NDArray

from .._typing import _Array1D, _IntArray
from ..atoms import Atoms
from ..kpoints import KPoints
from ..scf import SCF

def view(
    *args: Any,
    **kwargs: Any,
) -> Any: ...
def view_atoms(
    obj: Atoms | SCF,
    fods: NDArray[floating] | Sequence[NDArray[floating]] | None = ...,
    plot_n: bool = ...,
    percent: float = ...,
    isovalue: float | None = ...,
    surfaces: int = ...,
    size: _IntArray = ...,
) -> Any: ...
def view_contour(
    obj: Atoms | SCF,
    field: NDArray[floating] | None,
    axis: int = ...,
    value: float = ...,
    lines: int = ...,
    limits: _Array1D = ...,
    zoom: float = ...,
    linewidth: float = ...,
    size: _IntArray = ...,
) -> Any: ...
def view_file(
    filename: str | Sequence[str],
    isovalue: float = ...,
    gui: bool = ...,
    elec_symbols: Sequence[str] = ...,
    size: _IntArray = ...,
    **kwargs: Any,
) -> Any: ...
def executed_in_notebook() -> bool: ...
def plot_bandstructure(
    scf: SCF,
    spin: int | _IntArray = ...,
    size: _IntArray = ...,
) -> Any: ...
def plot_dos(
    scf: SCF,
    spin: int | _IntArray = ...,
    size: _IntArray = ...,
    **kwargs: Any,
) -> Any: ...
def view_kpts(
    kpts: KPoints,
    path: bool = ...,
    special: bool = ...,
    connect: bool = ...,
    size: _IntArray = ...,
) -> Any: ...
