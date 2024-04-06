# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import float64
from numpy.typing import NDArray

@overload
def libxc_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: None,
    tau: None,
) -> tuple[NDArray[float64], NDArray[float64], None, None]: ...
@overload
def libxc_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: NDArray[float64],
    tau: None,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], None]: ...
@overload
def libxc_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: NDArray[float64],
    tau: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]: ...
@overload
def pyscf_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: None,
    tau: None,
) -> tuple[NDArray[float64], NDArray[float64], None, None]: ...
@overload
def pyscf_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: NDArray[float64],
    tau: None,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], None]: ...
@overload
def pyscf_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: NDArray[float64],
    tau: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]: ...
