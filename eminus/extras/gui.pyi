# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, TypeAlias

from numpy import floating, integer
from numpy.typing import NDArray

from eminus.atoms import Atoms
from eminus.kpoints import KPoints
from eminus.scf import SCF

_Int: TypeAlias = integer[Any]
_Float: TypeAlias = floating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]
_IntArray: TypeAlias = Sequence[int] | NDArray[_Int]

def view(
    *args: Any,
    **kwargs: Any,
) -> Any: ...
def view_atoms(
    obj: Atoms | SCF,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    plot_n: bool | _ArrayReal = ...,
    percent: float = ...,
    isovalue: float | None = ...,
    surfaces: int = ...,
    size: _IntArray = ...,
) -> Any: ...
def view_contour(
    obj: Atoms | SCF,
    field: _ArrayReal | None,
    axis: int = ...,
    value: float = ...,
    lines: int = ...,
    limits: Sequence[float] | NDArray[_Float] = ...,
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
