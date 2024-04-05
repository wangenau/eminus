# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .energies import Energy
from .gth import GTH
from .kpoints import KPoints
from .logger import CustomLogger
from .typing import Array1D

class SCF:
    log: CustomLogger
    etol: float
    gradtol: float | None
    sic: bool
    disp: bool | dict[str, bool | str | None]
    smear_update: int
    energies: Energy
    is_converged: bool
    gth: GTH
    Vloc: NDArray[np.complex128]
    W: list[NDArray[np.complex128]] | None
    Y: list[NDArray[np.complex128]] | None
    Z: list[NDArray[np.complex128]] | None
    D: list[NDArray[np.complex128]] | None
    n: NDArray[np.float64] | None
    n_spin: NDArray[np.float64] | None
    dn_spin: NDArray[np.float64] | None
    tau: NDArray[np.float64] | None
    phi: NDArray[np.float64] | None
    exc: NDArray[np.float64] | None
    vxc: NDArray[np.complex128] | None
    vsigma: NDArray[np.complex128] | None
    vtau: NDArray[np.complex128] | None
    def __init__(
        self,
        atoms: Atoms,
        xc: str = ...,
        pot: str = ...,
        guess: str = ...,
        etol: float = ...,
        gradtol: float | None = ...,
        sic: bool = ...,
        disp: bool | dict[str, bool | str | None] = ...,
        opt: dict[str, int] | None = ...,
        verbose: int | str | None = ...,
    ) -> None: ...
    @property
    def atoms(self) -> Atoms: ...
    @atoms.setter
    def atoms(self, value: Atoms) -> None: ...
    @property
    def xc(self) -> list[str]: ...
    @xc.setter
    def xc(self, value: str | Sequence[str]) -> None: ...
    @property
    def pot(self) -> str: ...
    @pot.setter
    def pot(self, value: str) -> None: ...
    @property
    def guess(self) -> str: ...
    @guess.setter
    def guess(self, value: str) -> None: ...
    @property
    def opt(self) -> dict[str, int]: ...
    @opt.setter
    def opt(self, value: dict[str, int]) -> None: ...
    @property
    def verbose(self) -> str: ...
    @verbose.setter
    def verbose(self, value: int | str | None) -> None: ...
    @property
    def kpts(self) -> KPoints: ...
    @property
    def psp(self) -> str: ...
    @property
    def symmetric(self) -> bool: ...
    @property
    def xc_type(self) -> str: ...
    def run(self, **kwargs: Any) -> float: ...
    kernel = run
    def converge_bands(self, **kwargs: Any) -> SCF: ...
    def converge_empty_bands(
        self,
        Nempty: int | None = ...,
        **kwargs: Any,
    ) -> SCF: ...
    def recenter(self, center: float | Array1D | None = ...) -> SCF: ...
    def clear(self) -> SCF: ...

class RSCF(SCF):
    @property
    def atoms(self) -> Atoms: ...
    @atoms.setter
    def atoms(self, value: Atoms) -> None: ...

class USCF(SCF):
    @property
    def atoms(self) -> Atoms: ...
    @atoms.setter
    def atoms(self, value: Atoms) -> None: ...
