# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import overload

import numpy as np
from numpy.typing import NDArray

@overload
def libxc_functional(
    xc: str,
    n_spin: NDArray[np.float64],
    Nspin: int,
    dn_spin: None,
    tau: None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None, None]: ...
@overload
def libxc_functional(
    xc: str,
    n_spin: NDArray[np.float64],
    Nspin: int,
    dn_spin: NDArray[np.float64],
    tau: None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], None]: ...
@overload
def libxc_functional(
    xc: str,
    n_spin: NDArray[np.float64],
    Nspin: int,
    dn_spin: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def pyscf_functional(
    xc: str,
    n_spin: NDArray[np.float64],
    Nspin: int,
    dn_spin: None,
    tau: None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None, None]: ...
@overload
def pyscf_functional(
    xc: str,
    n_spin: NDArray[np.float64],
    Nspin: int,
    dn_spin: NDArray[np.float64],
    tau: None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], None]: ...
@overload
def pyscf_functional(
    xc: str,
    n_spin: NDArray[np.float64],
    Nspin: int,
    dn_spin: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
